# main.py
# Goal: Stitch the entire pipeline together
#        - lazy loading
#        - preprocessing
#        - streaming
#        - system monitoring

from pipeline.loader import load_image_paths
from pipeline.stream import image_stream
from utils.monitor import system_stats, FPSCounter

#  Path to your data folder
IMAGE_DIR = "data/images"

#  Step 1: Load image paths (no images yet)
paths = load_image_paths(IMAGE_DIR)

#  Step 2: Create a lazy image generator
stream = image_stream(paths)

#  Step 3: Setup FPS and memory tracking
fps = FPSCounter()
fps.start_timer()  # Students must remember to start the timer!

#  Step 4: Main loop — simulate edge deployment
for img in stream:
    # Here’s where inference would normally happen
    # e.g., output = model(img)

    # Track performance
    fps_val = fps.update()
    mem = system_stats()

    # Optimization: Reduce print frequency to save I/O overhead
    if fps.frames % 10 == 0:
        print(f"Frame processed | FPS: {fps_val:.2f} | Mem(MB): {mem:.2f}\r", end="")

    # TODO (Optional):
    # - Display image using OpenCV
    # - Save output to file
    # - Add a stop condition (e.g., break after N frames)