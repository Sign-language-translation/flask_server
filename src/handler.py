import os
import base64
import json
import tempfile
import shutil
import traceback
import pickle

import cv2
from runpod.serverless import start
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# from utils.test_mediapipe import extract_motion_data, motion_data_to_json
# from models.classify_attn import classify_json_file

# ************ CONVERT JSON *****************
import os
import json
import numpy as np

# Constants
MAX_FRAMES = 150
FACTOR = 1

# Functions (as provided in your code)
def create_feature_vector(frames_data, max_frames=MAX_FRAMES, factor=FACTOR):
    avg_frames = max_frames // factor
    feature_matrix = np.zeros((avg_frames, 75, 3), dtype=np.float32)

    for i in range(avg_frames):
        start_idx = i * factor
        feature_matrix[i] = average_frames(frames_data, start_idx, factor)

    return feature_matrix

def extract_features(frame):
    vector = []

    # Process pose features
    pose_features = frame.get('pose', [])
    for feature in pose_features:
        x = feature.get('x', 0.0)
        y = feature.get('y', 0.0)
        z = feature.get('z', 0.0)
        vector.append(np.array([x, y, z]))

    # Process hand features
    hands = frame.get('hands', [])
    for hand_index in range(2):  # Ensure exactly two hands (or placeholders)
        if hand_index < len(hands):
            hand_features = hands[hand_index]
            for feature in hand_features:
                x = feature.get('x', 0.0)
                y = feature.get('y', 0.0)
                z = feature.get('z', 0.0)
                vector.append(np.array([x, y, z]))
        else:
            # Add placeholder for missing hand
            vector.extend([np.array([0.0, 0.0, 0.0])] * 21)

    return vector

def average_frames(frames, start_idx, factor=FACTOR):
    """
    Averages FACTOR frames starting from start_idx.

    Args:
        frames (list): List of frame data.
        start_idx (int): The starting index for averaging.
        factor (int): The number of frames to average.

    Returns:
        np.ndarray: Averaged feature array of shape (75, 3).
    """
    end_idx = start_idx + factor
    vector = np.zeros((75, 3), dtype=np.float32)
    count = 0

    for i in range(start_idx, end_idx):
        if i >= len(frames):  # Stop if exceeding available frames
            break
        frame_vector = extract_features(frames[i])
        vector += frame_vector
        count += 1

    # Return the averaged vector
    return vector / count if count > 0 else np.zeros((75, 3), dtype=np.float32)


# json_path = "motion_data_old/brother.json"
#
# with open(json_path, "r") as file:
#     frames_data = json.load(file)
#
# # Create the feature vector
# feature_matrix = create_feature_vector(frames_data)
#
# # Print the shape and a sample of the matrix
# print(f"Feature matrix shape: {feature_matrix.shape}")
# print(f"Sample data:\n{feature_matrix[:2]}")  # Print first 2 averaged frames for verification


# ************* TEST MEDIAPIPE *****************************
import cv2
import os
import mediapipe as mp
import json
import numpy as np

# Global configuration
video_folder = "resources/sign_language_videos/"
json_folder = "resources/motion_data/"
output_folder = "resources/generated_videos/"


def extract_motion_data(video_name, folder_name=video_folder):
    # Initialize MediaPipe pose and hands modules
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Load video
    if not folder_name.endswith('/'):
        folder_name += '/'
    video_path = folder_name + video_name + ".mp4"
    cap = cv2.VideoCapture(video_path)

    # Output file for motion data
    output_data = []

    # Initialize MediaPipe
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose, \
            mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (required by MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process pose
            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)

            # Extract key points
            frame_data = {"pose": [], "hands": []}
            if pose_results.pose_landmarks:
                frame_data["pose"] = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    } for lm in pose_results.pose_landmarks.landmark
                ]

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    frame_data["hands"].append([
                        {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark
                    ])

            # Append to output data
            output_data.append(frame_data)

            # Optionally, draw landmarks on the frame for visualization
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show the frame
            # cv2.imshow('Sign Language Video', frame)  # uncomment to visulise the original video
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return output_data

    # Trim dead time using the modified detect_motion_and_trim
    # trimmed_data = detect_motion_and_trim(output_data)
    #
    # return trimmed_data


def motion_data_to_json(frames_data, video_name, folder_name, log_folder_path=None):
    # Ensure log folder exists
    if log_folder_path is None:
        log_folder_path = os.getcwd()  # Set log folder path to the current working directory

    os.makedirs(log_folder_path, exist_ok=True)

    # Path to log file
    log_path = os.path.join(log_folder_path, "defective_json_log.txt")

    # Check if motion data is empty
    if not frames_data:
        # Append the defective video name to the log file
        with open(log_path, "a") as log_file:
            log_file.write(video_name + ".json\n")

        print(f"Empty motion data for '{video_name}', logged to {log_path}")
        return  # Skip saving the empty JSON file

    # Save motion data to a file
    json_path = os.path.join(folder_name, video_name + ".json")
    json_path = json_path.replace("\\", "/")
    with open(json_path, "w") as f:
        json.dump(frames_data, f)

    print(f"Motion data saved to {json_path}")


def visualize_motion_data(video_name, json_folder):
    """Visualize motion data from a JSON file."""
    # Load motion data
    json_path = json_folder + video_name + ".json"
    with open(json_path, "r") as f:
        motion_data = json.load(f)

    # Create a blank canvas for visualization
    canvas_size = (720, 1280, 3)  # Height, Width, Channels

    for frame_data in motion_data:
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # White background

        # Draw pose landmarks
        if frame_data["pose"]:
            for lm in frame_data["pose"]:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

        # Draw hand landmarks
        for hand_landmarks in frame_data["hands"]:
            for lm in hand_landmarks:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

        # Show frame
        cv2.imshow('Visualizing Motion Data', canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Adjust delay as needed
            break

    cv2.destroyAllWindows()


def visualize_as_stick_figure(video_name):
    """Visualize motion data as a stick figure."""
    # Load motion data
    json_path = json_folder + video_name + ".json"
    with open(json_path, "r") as f:
        motion_data = json.load(f)

    # Create a blank canvas for visualization
    canvas_size = (720, 1280, 3)  # Height, Width, Channels

    # Define connections for stick figure (based on MediaPipe connections)
    pose_connections = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 12),  # Shoulders
        (23, 24),  # Hips
        (11, 23), (12, 24),  # Torso
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28)  # Right leg
    ]

    for frame_data in motion_data:
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # White background

        # Draw pose landmarks and connections
        if frame_data["pose"]:
            landmarks = frame_data["pose"]
            for start, end in pose_connections:
                if start < len(landmarks) and end < len(landmarks):
                    x1, y1 = int(landmarks[start]["x"] * canvas_size[1]), int(landmarks[start]["y"] * canvas_size[0])
                    x2, y2 = int(landmarks[end]["x"] * canvas_size[1]), int(landmarks[end]["y"] * canvas_size[0])
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw connections

            # Draw individual points
            for lm in landmarks:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

        # Show frame
        cv2.imshow('Stick Figure Animation', canvas)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Adjust delay as needed
            break

    cv2.destroyAllWindows()


def save_visualization_as_video(video_name):
    """Save the visualization as a video file."""
    # Load motion data
    json_path = json_folder + video_name + ".json"
    with open(json_path, "r") as f:
        motion_data = json.load(f)

    # Create a blank canvas for visualization
    canvas_size = (720, 1280, 3)  # Height, Width, Channels

    # Initialize video writer
    output_path = output_folder + video_name + "_recreated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10  # Adjust frames per second
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_size[1], canvas_size[0]))

    for frame_data in motion_data:
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # White background

        # Draw pose landmarks
        if frame_data["pose"]:
            for lm in frame_data["pose"]:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

        # Draw hand landmarks
        for hand_landmarks in frame_data["hands"]:
            for lm in hand_landmarks:
                x, y = int(lm["x"] * canvas_size[1]), int(lm["y"] * canvas_size[0])
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

        # Write the frame to the video
        out.write(canvas)

    out.release()
    print(f"Recreated video saved to {output_path}")


# def json_to_numpy(json_file_path):
#     """
#     Load JSON data from a file and convert it to a NumPy array.

#     Args:
#         json_file_path (str): The path to the JSON file.

#     Returns:
#         np.ndarray: A NumPy array representing the motion data, or None if an error occurs.
#     """
#     try:
#         # Step 1: Load the JSON data from the file
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             frames_data = json.load(f)

#         if not frames_data:
#             raise ValueError("The JSON file is empty or invalid.")

#         # Step 2: Convert JSON data to a feature vector using create_feature_vector
#         feature_vector = create_feature_vector(frames_data)

#         print(f"Successfully converted JSON file '{json_file_path}' to NumPy array.")
#         return feature_vector

#     except Exception as e:
#         print(f"Failed to convert JSON file '{json_file_path}' to NumPy array: {e}")
#         return None


# if __name__ == "__main__":
#     for video in ["brother"  # Replace with the name of your video (without extension)
#
#     # # Extract motion data from video
#     extract_motion_data(video_name)
#     #
#     # # Visualize motion data
#     visualize_motion_data(video_name)
#     #
#     # # Save visualization as video
#     # save_visualization_as_video(video_name)
#
#     # visualize_as_stick_figure(video_name)

existing_words = ["help"]


def create_original_motion_data(folder_name="resources/sign_language_videos",
                                output_folder_path="resources/motion_data"):
    # Iterate through all files in the given folder
    for file_name in os.listdir(folder_name):
        # Check if the file is a video file (e.g., .mp4, .avi, etc.)
        if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            if any(file_name.startswith(word + "_") for word in existing_words):
                # Remove the file extension to get the video name
                video_name = os.path.splitext(file_name)[0]

                # Extract motion data from the video
                trim_data = extract_motion_data(video_name, folder_name=folder_name)
                motion_data_to_json(trim_data, video_name, output_folder_path)


# ******************** CLASSIFY ATTN *************

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, InputSpec
# from utils.conver_json_to_vector import create_feature_vector
# from utils.test_mediapipe import extract_motion_data, motion_data_to_json

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



# ************************************************


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/model-5_14000_vpw.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "models/label_encoder_model-5_14000_vpw.pkl")
print(f"🔍 MODEL_PATH: {MODEL_PATH}")
print(f"🔍 ENCODER_PATH: {ENCODER_PATH}")
# Load label encoder
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
label_classes = list(label_encoder.classes_)



def handler(event):
    try:
        print("🔍 Event received.")
        input_data = event['input']
        filename = input_data.get("filename", "video.mp4")
        base64_video = input_data["content"]
        start_sec, end_sec = input_data["tuple"]
        print(f"🔍 Input parsed: filename={filename}, start={start_sec}, end={end_sec}")

        # Step 1: Save temp video
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, filename)
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(base64_video))
        print(f"🔍 Video saved at {video_path}")

        # Step 2: Trim + extract motion
        segment_path = cut_segment(video_path, start_sec, end_sec)
        print(f"🔍 Segment cut: {segment_path}")

        base_name = os.path.splitext(os.path.basename(segment_path))[0]
        motion = extract_motion_data(base_name, folder_name=os.path.dirname(segment_path))
        print(f"🔍 Motion data extracted.")

        motion_data_to_json(motion, base_name, folder_name=os.path.dirname(segment_path))
        print(f"🔍 Motion data saved to JSON.")

        json_path = os.path.join(os.path.dirname(segment_path), f"{base_name}.json")
        with open(json_path, "r", encoding="utf-8") as jf:
            motion_json = json.load(jf)
        print(f"🔍 JSON file loaded from {json_path}")

        # Step 3: Predict
        prediction = classify_json_file(MODEL_PATH, motion_json, label_classes)
        print(f"✅ Prediction successful: {prediction}")

        return {"prediction": prediction}

    except Exception as e:
        print("❌ An exception occurred:")
        traceback.print_exc()
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Temp directory {temp_dir} removed.")


def cut_segment(video_path, start_sec, end_sec):
    output_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🔍 Cutting segment from {start_sec}s to {end_sec}s with FPS={fps}")
    if fps == 0:
        raise ValueError("Invalid FPS")

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_dir, f"segment_{start_sec}_{end_sec}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return output_path


# if __name__ == "__main__":
#     print("🚀 Starting RunPod serverless test handler...")
start({"handler": handler})  # Required by RunPod serverless


#**************************** BASIC SERVER ********************************
# import runpod
# import time
#
#
# def handler(event):
#     #   This function processes incoming requests to your Serverless endpoint.
#     #
#     #    Args:
#     #        event (dict): Contains the input data and request metadata
#     #
#     #    Returns:
#     #       Any: The result to be returned to the client
#
#     # Extract input data
#     print(f"Worker Start")
#     input = event['input']
#
#     prompt = input.get('prompt')
#     seconds = input.get('seconds', 0)
#
#     print(f"Received prompt: {prompt}")
#     print(f"Sleeping for {seconds} seconds...")
#
#     # You can replace this sleep call with your own Python code
#     time.sleep(seconds)
#
#     return prompt
#
#
# # Start the Serverless function when the script is run
# if __name__ == '__main__':
#     runpod.serverless.start({'handler': handler})
#
#
#**************************** REQ ********************************

# absl-py==2.3.0
# astunparse==1.6.3
# attrs==25.3.0
# blinker==1.9.0
# cachetools==5.5.2
# certifi==2025.1.31
# cffi==1.17.1
# charset-normalizer==3.4.2
# click==8.2.1
# contourpy==1.3.2
# cycler==0.12.1
# Flask==3.1.1
# flask-cors==6.0.0
# flatbuffers==25.2.10
# fonttools==4.58.1
# gast==0.4.0
# google-auth==2.40.2
# google-auth-oauthlib==1.0.0
# google-pasta==0.2.0
# grpcio==1.71.0
# h5py==3.13.0
# idna==3.10
# itsdangerous==2.2.0
# Jinja2==3.1.6
# keras==2.12.0
# kiwisolver==1.4.8
# libclang==18.1.1
# ll==1.0
# Markdown==3.8
# markdown-it-py==3.0.0
# MarkupSafe==3.0.2
# matplotlib==3.10.3
# mdurl==0.1.2
# mediapipe==0.10.21
# ml-dtypes==0.3.2
# namex==0.1.0
# numpy==1.23.5
# oauthlib==3.2.2
# opt_einsum==3.4.0
# optree==0.16.0
# packaging==25.0
# pillow==11.2.1
# protobuf==4.25.8
# pyasn1==0.6.1
# pyasn1_modules==0.4.2
# pycparser==2.22
# Pygments==2.19.1
# pyparsing==3.2.3
# python-dateutil==2.9.0.post0
# requests==2.32.3
# requests-oauthlib==2.0.0
# rich==14.0.0
# rsa==4.9.1
# scipy==1.15.3
# sentencepiece==0.2.0
# six==1.17.0
# sounddevice==0.5.2
# tensorboard==2.12.3
# tensorboard-data-server==0.7.2
# tensorflow-estimator==2.12.0
# tensorflow-io-gcs-filesystem==0.37.1
# tensorflow==2.12.0
# termcolor==3.1.0
# typing_extensions==4.13.2
# urllib3==2.4.0
# Werkzeug==3.1.3
# wrapt==1.14.1
# opencv-python-headless==4.9.0.80
# moviepy==1.0.3
# imageio==2.37.0
# imageio-ffmpeg==0.6.0
# python-dotenv==1.0.1
# decorator==4.4.2
# tqdm==4.67.1
# runpod