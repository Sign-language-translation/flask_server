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

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.test_mediapipe import extract_motion_data, motion_data_to_json
from models.classify_attn import classify_json_file

# Load paths
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