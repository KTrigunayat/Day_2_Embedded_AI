# preprocess.py
# Responsibility:
# 1. Take ONE image already loaded into memory
# 2. Apply minimal preprocessing for inference
# 3. Return a processed image ready for a model

import cv2
import numpy as np


def preprocess_image(img, size=(224, 224)):
    """
    Preprocess a single image for edge inference.

    Constraints (IMPORTANT):
    - Resize BEFORE normalization
    - Use float32 (not float64)
    - Avoid unnecessary memory copies

    Think:
    - Why resize first?
    - Why not normalize uint8 directly?
    - Why float32 instead of default NumPy float?
    """

    # 1: Resize the image to the given size
    # Hint: OpenCV expects (width, height)
    # img = ?
    img = cv2.resize(img, size)

    # 2: Convert image to float32
    # Hint: NumPy dtype conversion
    # img = ?
    img = img.astype(np.float32)

    # 3: Normalize pixel values to range [0, 1]
    # IMPORTANT: Do this IN-PLACE if possible
    # Optimization: Multiplication is slightly faster than division
    img *= 0.00392156862745098  # 1/255.0

    return img