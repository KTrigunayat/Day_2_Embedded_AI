# main.py
# Full edge-camera pipeline orchestration
# - Capture from webcam
# - Sample frames at controlled FPS
# - Preprocess each frame
# - Simulate model inference
# - Track performance

from camera.webcam import Webcam
from pipeline.sampler import FrameSampler
from pipeline.preprocess import preprocess
from inference.dummy_model import dummy_inference
from utils.metrics import Monitor
import time


#  Initialize modules
cam = Webcam()
sampler = FrameSampler(target_fps=5)
monitor = Monitor()
monitor.start = time.time()  # STUDENTS MUST DO THIS

try:
    while True:
        # Step 1: Grab a frame
        frame = cam.read()
        if frame is None:
            continue  # Handle camera disconnect gracefully

        # Step 2: Check if we should process this frame
        if not sampler.allow():
            continue  # Drop frame intentionally

        # Step 3: Preprocess the image for inference
        processed = preprocess(frame)

        # Step 4: Simulate inference (add your model later)
        _ = dummy_inference(processed)

        # Step 5: Monitor performance
        fps, mem = monitor.update()
        print(f"FPS: {fps:.2f} | Memory: {mem:.2f} MB")

except KeyboardInterrupt:
    # Cleanup
    cam.release()
    print("Camera released. Pipeline stopped.")