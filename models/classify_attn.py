import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, InputSpec
from utils.conver_json_to_vector import create_feature_vector
from utils.test_mediapipe import extract_motion_data, motion_data_to_json

# Global model cache
MODEL_CACHE = None

# ────────────────────────────────────────────────────────────────────────────────
# 1) Re-declare your custom SelfAttention so load_model can deserialize it
# ────────────────────────────────────────────────────────────────────────────────
class SelfAttention(Layer):
    """
    Simple self-attention over time dimension.
    Input shape: (batch, time, features)
    Output shape: (batch, features)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        F = input_shape[-1]  # features
        self.W = self.add_weight(
            name='W_attn', shape=(F, F),
            initializer='glorot_uniform', trainable=True
        )
        self.b = self.add_weight(
            name='b_attn', shape=(F,),
            initializer='zeros', trainable=True
        )
        self.u = self.add_weight(
            name='u_attn', shape=(F,),
            initializer='glorot_uniform', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, T, F)
        u_it = tf.tanh(tf.tensordot(inputs, self.W, axes=[2,0]) + self.b)  # (b, T, F)
        scores = tf.tensordot(u_it, self.u, axes=[2,0])                   # (b, T)
        alphas = tf.nn.softmax(scores, axis=1)                             # (b, T)
        context = tf.matmul(tf.expand_dims(alphas, 1), inputs)            # (b, 1, F)
        return tf.squeeze(context, 1)                                      # (b, F)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# ────────────────────────────────────────────────────────────────────────────────
# 2) File / label loading helpers
# ────────────────────────────────────────────────────────────────────────────────
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    return None

def load_label_mapping(file_path):
    with open(file_path, 'rb') as f:
        le = pickle.load(f)
    print(f"Label encoder loaded from {file_path}")
    return list(le.classes_)

# ────────────────────────────────────────────────────────────────────────────────
# 3) The core classify_json_file now loads with custom_objects
# ────────────────────────────────────────────────────────────────────────────────
def classify_json_file(model_filename, json_content, label_mapping):
    # 1) load model once per call (or cache externally)
    # model = load_model(
    #     model_filename,
    #     compile=False,
    #     custom_objects={'SelfAttention': SelfAttention}
    # )
    global MODEL_CACHE
    if MODEL_CACHE is None:
        MODEL_CACHE = load_model(
            model_filename,
            compile=False,
            custom_objects={'SelfAttention': SelfAttention}
        )
    model = MODEL_CACHE

    # 2) convert JSON → feature array
    mat = create_feature_vector(json_content)  # e.g. shape (T,H,W,C)
    # add batch dim
    x = np.expand_dims(mat, 0)                 # shape (1, T, H, W, C)
    # 3) inference
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=-1)[0])
    return label_mapping[idx]



# ────────────────────────────────────────────────────────────────────────────────
# 4) Your existing classify / classify_single_word flow, unchanged
# ────────────────────────────────────────────────────────────────────────────────
def classify(input_folder_path_of_videos, temp_folder_path_of_jsons, model_file_path, label_encoder_file_path):
    for file_name in os.listdir(input_folder_path_of_videos):
        if file_name.lower().endswith('.mp4'):
            classify_single_word(
                input_folder_path_of_videos + "/" + file_name,
                input_folder_path_of_videos,
                temp_folder_path_of_jsons,
                model_file_path,
                label_encoder_file_path
            )

def classify_single_word(input_video_folder_name, video_file_name, temp_folder_path_of_jsons, model_file_path, label_encoder_file_path):
    file_path = os.path.join(input_video_folder_name, video_file_name)
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return None

    if video_file_name.lower().endswith(".mp4"):
        file_base = os.path.splitext(video_file_name)[0]

        # your mediapipe steps:
        trim_data = extract_motion_data(file_base, folder_name=input_video_folder_name)
        motion_data_to_json(trim_data, file_base, folder_name=temp_folder_path_of_jsons)

        json_path = f"{temp_folder_path_of_jsons}/{file_base}.json"
        json_content = read_json_file(json_path)
        if json_content is None:
            return None

        labels = load_label_mapping(label_encoder_file_path)
        prediction = classify_json_file(model_file_path, json_content, labels)
        print(f"Real label: {file_base}, Predicted Label: {prediction}\njson_len: {len(json_content)}")
        return prediction

    return None

# ────────────────────────────────────────────────────────────────────────────────
# 5) Example usage stub — adjust paths if needed
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_folder_path_of_videos  = 'single_word_videos/videos'
    temp_folder_path_of_jsons    = 'single_word_videos/jsons'
    model_file_name              = 'models/local_models/3d_rnn_cnn_attn_on_70_vpw.keras'
    label_encoder_file_name      = 'models/local_models/label_encoder_attn_70_vpw.pkl'

    classify(
        input_folder_path_of_videos,
        temp_folder_path_of_jsons,
        model_file_name,
        label_encoder_file_name
    )