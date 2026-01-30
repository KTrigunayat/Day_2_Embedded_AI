# monitor.py
# Responsibility:
# 1. Track memory used
# 2. Track how fast (FPS) we're processing

import psutil
import time


def system_stats():
    """
    Return current system memory usage in MB.

    Think:
    - Why do we care about memory on Jetson?
    - What happens if we cross 3.5 GB RAM?
    """

    # 1. Use psutil to get virtual memory
    # 2. Return memory used in megabytes (MB)
    mem = psutil.virtual_memory()
    used_mb = mem.used / (1024 * 1024)
    return used_mb


class FPSCounter:
    def __init__(self):
        """
        Initialize timer and frame counter.
        Think:
        - Why use elapsed time instead of counting per second?
        """
        self.start = None
        self.frames = 0

    def start_timer(self):
        # Record the current time as start
        self.start = time.time()

    def update(self):
        # 1. Increment frame count
        # 2. Compute time elapsed
        # 3. Return FPS (frames / elapsed time)
        self.frames += 1
        if self.start is None:
            return 0.0
        
        elapsed = time.time() - self.start
        if elapsed > 0:
            return self.frames / elapsed
        return 0.0