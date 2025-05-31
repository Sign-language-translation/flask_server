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