# dummy_model.py
# Responsibility:
# - Simulate the latency and structure of a real model
# - Let us test the end-to-end pipeline without ML distractions

import time
import numpy as np


def dummy_inference(frame):
    """
    Simulate a model's processing time.

    Think:
    - We assume this function is called on every frame that passes the sampler
    - A real model might take 20–50 ms to process a frame

    Your job:
    - Pause execution briefly to simulate inference load
    - Return a fake prediction (mean pixel value is fine)

    Why?
    - This lets us test FPS, RAM, flow, etc. without needing a real model.
    """

    # 1. Use time.sleep() to simulate delay (~30ms)
    time.sleep(0.03)

    # 2. Return a fake value — e.g., the average brightness of the frame
    if frame is not None:
        return np.mean(frame)

    return 0.0  # placeholder prediction