# predictor.py

import numpy as np
import tensorflow as tf
import pickle
import json
# from utils.conver_json_to_vector import create_feature_vector
from utils.conver_json_to_vector import create_feature_vector
from tensorflow.keras.models import load_model

# Load trained model and label encoder
model = tf.keras.models.load_model("models/best_so_far/model 5/model-5_14000_vpw.keras", custom_objects={"SelfAttention": __import__("models/best_so_far/model 5/model_5").SelfAttention})
with open("label_encoder_model-5_14000_vpw.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# def predict_from_json(json_data):
#     """
#     Input: JSON (dict) with video keypoints
#     Output: predicted label (str)
#     """
#     try:
#         # # Convert JSON to feature vector
#         # vec = create_feature_vector(json_data)  # shape: (T, H, W, C)
#         # vec = np.expand_dims(vec, axis=0)       # shape: (1, T, H, W, C)

#         # Predict
#         prediction = model.predict(json_data)
#         predicted_class = prediction.argmax(axis=1)[0]
#         label = label_encoder.inverse_transform([predicted_class])[0]

#         return label
#     except Exception as e:
#         return f"Prediction failed: {str(e)}"

def classify_json_file(json_content, model_filename , label_mapping):
    """
    Classifies a dummy matrix using a pre-trained model.

    Args:
        model_filename (str): Path to the saved model.
        input_shape (tuple): Shape of the input data (e.g., (50, 75, 3)).
        label_mapping (dict): Mapping of class indices to labels.

    Returns:
        str: Predicted class label.
    """
    # Load the saved model
    model = load_model(model_filename)

    input_matrix = create_feature_vector(json_content)
    # Assuming `input_data` is your input of shape (32, 75, 3)
    reshaped_data = input_matrix.reshape((1, 50, 75, 3))  # Add batch and time steps dimensions

    # Pass reshaped_data to the model
    predictions = model.predict(reshaped_data)
    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the predicted class index

    # Map the predicted index to the corresponding label
    predicted_label = label_mapping[predicted_class]

    return predicted_label