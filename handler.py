# import os
# import base64
# import json
# import tempfile
# import shutil
# import traceback
# import pickle
#
# import cv2
from runpod.serverless import start
#
# from utils.test_mediapipe import extract_motion_data, motion_data_to_json
# from models.classify_attn import classify_json_file
#
# # Load paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models/model-5_14000_vpw.keras")
# ENCODER_PATH = os.path.join(BASE_DIR, "models/label_encoder_model-5_14000_vpw.pkl")
# print(f"🔍 MODEL_PATH: {MODEL_PATH}")
# print(f"🔍 ENCODER_PATH: {ENCODER_PATH}")
# # Load label encoder
# with open(ENCODER_PATH, 'rb') as f:
#     label_encoder = pickle.load(f)
# label_classes = list(label_encoder.classes_)
#
#
#
# def handler(event):
#     try:
#         print("🔍 Event received.")
#         input_data = event['input']
#         filename = input_data.get("filename", "video.mp4")
#         base64_video = input_data["content"]
#         start_sec, end_sec = input_data["tuple"]
#         print(f"🔍 Input parsed: filename={filename}, start={start_sec}, end={end_sec}")
#
#         # Step 1: Save temp video
#         temp_dir = tempfile.mkdtemp()
#         video_path = os.path.join(temp_dir, filename)
#         with open(video_path, "wb") as f:
#             f.write(base64.b64decode(base64_video))
#         print(f"🔍 Video saved at {video_path}")
#
#         # Step 2: Trim + extract motion
#         segment_path = cut_segment(video_path, start_sec, end_sec)
#         print(f"🔍 Segment cut: {segment_path}")
#
#         base_name = os.path.splitext(os.path.basename(segment_path))[0]
#         motion = extract_motion_data(base_name, folder_name=os.path.dirname(segment_path))
#         print(f"🔍 Motion data extracted.")
#
#         motion_data_to_json(motion, base_name, folder_name=os.path.dirname(segment_path))
#         print(f"🔍 Motion data saved to JSON.")
#
#         json_path = os.path.join(os.path.dirname(segment_path), f"{base_name}.json")
#         with open(json_path, "r", encoding="utf-8") as jf:
#             motion_json = json.load(jf)
#         print(f"🔍 JSON file loaded from {json_path}")
#
#         # Step 3: Predict
#         prediction = classify_json_file(MODEL_PATH, motion_json, label_classes)
#         print(f"✅ Prediction successful: {prediction}")
#
#         return {"prediction": prediction}
#
#     except Exception as e:
#         print("❌ An exception occurred:")
#         traceback.print_exc()
#         return {
#             "error": str(e),
#             "trace": traceback.format_exc()
#         }
#     finally:
#         if 'temp_dir' in locals() and os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)
#             print(f"🧹 Temp directory {temp_dir} removed.")
#
#
# def cut_segment(video_path, start_sec, end_sec):
#     output_dir = tempfile.mkdtemp()
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"🔍 Cutting segment from {start_sec}s to {end_sec}s with FPS={fps}")
#     if fps == 0:
#         raise ValueError("Invalid FPS")
#
#     start_frame = int(start_sec * fps)
#     end_frame = int(end_sec * fps)
#
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_path = os.path.join(output_dir, f"segment_{start_sec}_{end_sec}.mp4")
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#     for _ in range(end_frame - start_frame):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
#
#     cap.release()
#     out.release()
#     return output_path
#
#
# if __name__ == "__main__":
#     # start({"handler": handler})  # Required by RunPod serverless
#     print("🚀 Starting RunPod serverless test handler...")
#     fake_event = {
#   "input": {
#     "filename": "video.mp4",
#     "content": "Xi/a2qVfiETuPMn2M0u6CwIvTKm3QSDC6PRo+9LtVCgcQMPS2mu/tYcUTJf/a2hVP22/2FDjF/p8KBzrXt9Pzte3/hva12/8wb112/+CTNTSrxI9yBvNS/bFar7riFfQX8IPci9DL1CY7+CIlcoPZX1VeeZ9eXWkCddCoVCy+F3dC313/4kRodD/QdPNQaTPTA73tT2CLsWuu39Dk+2fNQ1L99RE1DU5qGp5T+cPR7081DU360/71ER5q8lzETEfLDROzVPt+tznfT/yw1mlvTT2/biIV0tLqmntt/grXR611fuHmjRsqNH61pf3nj3R6MtOv4Uo0eWyEmvrX74KpCSB6BjMdZiz+Nm0N7WGWWUtf61VVb/+E0+zz2Kx9Mrrrkv/ErI0mY70LybS/M+64aJsddf+y0etCtqx2Ox+Cex3v+rhaDspmXkXixM19/pgou/feteyd/Y/3ovpagjpvvxXqJ69fgir3+FEvY7Ho15GF/q8zoH+FKrVVq1rrjnwS9dVVOq6D3r16RVnt/4JlqteLuGOhq0nr3/zOv4MOvrF6frgmsSFQ4wmNFXypflRN9dvT6/PWn/4Jr/OGb5g//wYUsW9iT8oJBnQn3EBKJGlDAcfTFvX/4pUftNLYl9rrRLXHxXW1+1Xwnrr8qF+q/zV07RUYRsC+vyrr13M10vNr3ylqq/a0L5a6XhbS9HXt+uFNH7DofysPr4lpa1/GKvpV9V+ya3zHX9LXZAlYv0Ll2o1BhY15V7k2Ltt/huG31Xb/4Q7Pmw2ZCCUjWDJlqs6xME9L6qdOPgo9aM1Pur6/JX6QKNde/nCHr1t284WBNKGA33rXwW+td+FIKuvpdJ+eI16/hOu0v2zVqhdOxTr/CS15uq5d/xrY+RBoWvYqEvRf4cS9bf/gs9Jf1eq1yOv79et1l9/LXuCcgTyouipr7lcJWvX8IV6+tdI3a84r0JiVtXyKv4p67f4j1rXlGV9Oxqv2LSFeq/xy6fS7H4Q6VJH6r4J10ORdi3+ESyBmxL791KEq9Cr0gh6119WXXNr+Wv3CW37VZTKzf5XZ66v2wxoHoH16f0V4R6+UN6AeGF/7TW01lXqRIul+/S8JVo0fWX2RZ7BNkU2tCHjxLSbCCvqwjBF7B9yMJSEH9cjBb66++EFr6rX499Usy9D/BL29A/vjaPW1WqxhMbBAUEfOv/MtfgvNsfoJevf2Ja9eMTF7KwPOv6Xgq0ZzXtdb/bX3jEavQlWsel7ygpJYux9Dz1cUXrV0vS7rVzNf0SvsI6J79b8EWv26CIsEvkX1+4YCIS2/v4S29f5taXjq/rX2hFV6Jfj9ehUKhHUvZ9W+HvWtdRpMLZWfrjzMKlHF+36ugTntSsx1M/vjBXsPQH2FWytXtguEm+uzv5SvQXy",
#     "tuple": [
#       0,
#       3
#         ]
#       }
#     }
#     print(handler(fake_event))
def handler(event):
    print("Received event:", event)  # This will appear in the RunPod Logs tab
    return {
        "status": "success",
        "input": event,
        "message": "Request processed successfully."
    }

