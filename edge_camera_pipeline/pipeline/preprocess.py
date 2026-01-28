# preprocess.py
# Responsibility:
# - Resize and normalize camera frames
# - Return model-ready float32 image

import cv2
import numpy as np


def preprocess(frame, size=(224, 224)):
    """
    Resize and normalize image for model input.

    Constraints:
    - Use float32
    - Normalize to [0, 1]
    - Resize FIRST, then normalize
    - Keep dimensions consistent (HxWxC → CHW if needed)

    Think:
    - Why float32 and not uint8?
    - Why resize before normalization?
    """

    if frame is None:
        return None

    # 1: Resize the frame to 'size' (e.g., 224x224)
    # Use OpenCV resize
    resized = cv2.resize(frame, size)

    # 2: Convert to float32
    # 3: Normalize pixels to range [0, 1]
    # In-place normalization to save memory
    normalized = resized.astype(np.float32)
    normalized /= 255.0

    return normalized  # final processed image