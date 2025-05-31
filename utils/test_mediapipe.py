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
        (11, 12),            # Shoulders
        (23, 24),            # Hips
        (11, 23), (12, 24),  # Torso
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28)   # Right leg
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

def create_original_motion_data(folder_name = "resources/sign_language_videos", output_folder_path = "resources/motion_data"):
    # Iterate through all files in the given folder
    for file_name in os.listdir(folder_name):
        # Check if the file is a video file (e.g., .mp4, .avi, etc.)
        if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            if any(file_name.startswith(word + "_") for word in existing_words):
                # Remove the file extension to get the video name
                video_name = os.path.splitext(file_name)[0]

                # Extract motion data from the video
                trim_data = extract_motion_data(video_name, folder_name = folder_name)
                motion_data_to_json(trim_data, video_name, output_folder_path)

if __name__ == "__main__":
    create_original_motion_data(folder_name = "resources/generated_videos", output_folder_path = "resources/original_motion_data")

