
import os
import socket
import struct
import json
import base64
import time
import numpy as np

SERVER_HOST = 'localhost'
SERVER_PORT = 5002
CONNECT_TIMEOUT = 5  # seconds

def encode_video_to_base64(path):
    with open(path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def prepare_video_payload(video_path, seg):
    return {
        "filename": os.path.basename(video_path),
        "content": encode_video_to_base64(video_path),
        "tuples": seg  # ✅ use 'tuples' to match Java-side field
    }

def send_video_payload(payload):
    try:
        s = socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=CONNECT_TIMEOUT)
        s.settimeout(120)
        print("✅ Connected successfully!")
    except socket.timeout:
        print(f"❌ Connection timed out after {CONNECT_TIMEOUT} seconds")
        return
    except OSError as e:
        print(f"❌ Connection failed: [Errno {e.errno}] {e.strerror}")
        return

    try:
        # Encode the single payload
        raw_str = json.dumps(payload)
        encoded = raw_str.encode('utf-8')
        s.sendall(struct.pack('!I', len(encoded)))  # Send 4-byte length
        s.sendall(encoded)  # Then JSON payload

        # Receive 4-byte response length
        header = s.recv(4)
        if len(header) < 4:
            print("❌ Incomplete response header")
            return
        total = struct.unpack('!I', header)[0]

        # Read full response
        data = b''
        while len(data) < total:
            chunk = s.recv(min(4096, total - len(data)))
            if not chunk:
                break
            data += chunk

        if len(data) < total:
            print("❌ Incomplete response body")
            return

        results = json.loads(data.decode('utf-8'))
        print("✅ Server Response:")
        for i, r in enumerate(results):
            print(f"{i+1}. {r}")

    except Exception as e:
        print(f"❌ Communication error: {e}")
    finally:
        s.close()
        return results


##########

def create_segments_list(video_duration):
    segments_list = []
    MIN_WINDOW_SEC = 1
    MAX_WINDOW_SEC = 2
    START_TIME_STEP_SEC = 0.15
    WINDOW_SIZE_STEP_SEC = 1


    for start_sec in np.arange(0, video_duration - MIN_WINDOW_SEC + 0.001, START_TIME_STEP_SEC):
        for segment_duration in np.arange(MIN_WINDOW_SEC, MAX_WINDOW_SEC + 0.001, WINDOW_SIZE_STEP_SEC):
            end_sec = start_sec + segment_duration

            # Trim segment to not go past video duration
            if start_sec >= video_duration:
                continue
            if end_sec > video_duration:
                end_sec = video_duration

            segments_list.append((start_sec, end_sec))

    return segments_list
###########
import cv2


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0:
        duration = total_frames / fps
        return round(duration, 2)
    else:
        raise ValueError("Invalid FPS value")


if __name__ == "__main__":
    start = time.time()

    video_path = "../test_codes_and_files/sentence_videos/yanoos_sentence.mp4"
    payload = prepare_video_payload(video_path, create_segments_list(get_video_duration(video_path)))

    send_video_payload(payload)

    end = time.time()
    print(f"✅ Sent 1 video | Time elapsed: {end - start:.2f} sec")