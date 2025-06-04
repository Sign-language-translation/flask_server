# import json
# import tempfile
# import shutil
# import base64
# import traceback
# import pickle
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#
# from flask import Flask, request, jsonify
# import cv2
#
# from utils.test_mediapipe import extract_motion_data, motion_data_to_json
# from models.classify_attn import classify_json_file
#
# app = Flask(__name__)
#
# # MODEL_PATH = "../models/model-5_14000_vpw.keras"
# # ENCODER_PATH = "../models/label_encoder_model-5_14000_vpw.pkl"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "../models/model-5_14000_vpw.keras")
# ENCODER_PATH = os.path.join(BASE_DIR, "../models/label_encoder_model-5_14000_vpw.pkl")
#
# @app.route('/health', methods=['GET'])
# def health():
#     return "ok", 200
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         filename = data.get("filename", "video.mp4")
#         file_base = os.path.splitext(filename)[0]
#         video_b64 = data["content"]
#         seg = data["tuple"]  # should be [start, end] pair
#
#         print(f"🛠️ Segment: {seg}")
#
#         # Step 1: Save base64 video to temp dir
#         temp_dir = tempfile.mkdtemp()
#         video_path = os.path.join(temp_dir, filename)
#         with open(video_path, "wb") as f:
#             f.write(base64.b64decode(video_b64.encode("utf-8")))
#
#         # Step 2: Cut segment from video and extract motion
#         segment_path, _, _ = cut_segments(video_path, seg[0], seg[1])
#         trim_data = extract_motion_data(os.path.splitext(os.path.basename(segment_path))[0], folder_name=os.path.dirname(segment_path))
#         motion_data_to_json(trim_data, os.path.splitext(os.path.basename(segment_path))[0], folder_name=os.path.dirname(segment_path))
#
#         json_path = os.path.join(os.path.dirname(segment_path), f"{os.path.splitext(os.path.basename(segment_path))[0]}.json")
#         with open(json_path, "r", encoding="utf-8") as jf:
#             motion_json = json.load(jf)
#
#         # Step 3: Run classification
#         labels = load_label_mapping(ENCODER_PATH)
#         prediction = classify_json_file(MODEL_PATH, motion_json, labels)
#
#         print(f"🎬 {filename} → {prediction}")
#
#         return prediction
#
#     except Exception as e:
#         print("❌ Exception during prediction:")
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if 'temp_dir' in locals() and os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)
#
#
# def cut_segments(video_path,start_sec,end_sec):
#     output_folder = tempfile.mkdtemp()
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if fps == 0:
#         cap.release()
#         raise ValueError("invalid video")
#     duration = total_frames / fps
#     duration = round(duration, 2)
#     start_frame = int(start_sec * fps)
#     end_frame = int(end_sec * fps)
#     segment_filename = f"segment_{start_sec}_{end_sec}.mp4"
#     segment_path = os.path.join(output_folder, segment_filename)
#
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
#
#     for _ in range(end_frame - start_frame):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
#
#     out.release()
#     return (segment_path, start_sec, end_sec)
#
#
# def load_label_mapping(file_path):
#     with open(file_path, 'rb') as f:
#         le = pickle.load(f)
#     print(f"Label encoder loaded from {file_path}")
#     return list(le.classes_)
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=6000, threaded=True)



import os
import base64
import json
import tempfile
import shutil
import traceback
import pickle

import cv2
from runpod.serverless import start, rp_handler

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

        return prediction

    except Exception as e:
        print("❌ An exception occurred:")
        traceback.print_exc()
        return {"error": str(e)}

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


if __name__ == "__main__":
    start({"handler": handler})  # Required by RunPod serverless