# sampler.py
# Responsibility:
# - Control which frames are allowed through
# - Drop extra frames to match a target FPS rate

import time


class FrameSampler:
    def __init__(self, target_fps=5):
        """
        Control frame rate by skipping frames.

        Example:
        - If camera gives 30 FPS, and target_fps = 5,
          we should allow only ~1 every 6 frames

        Think:
        - What happens if we try to process all frames?
        - Why does time.sleep() NOT help us here?
        """

        # 1. Compute time interval between allowed frames
        self.interval = 1.0 / target_fps if target_fps > 0 else 0
        
        # 2. Set up a variable to track last allowed timestamp
        self.last_time = 0

    def allow(self):
        """
        Returns True if enough time has passed since last allowed frame.
        Otherwise, returns False (drop the frame).

        Think:
        - Why use time.time() instead of counting frames?
        - Why reset last_time only when allowing?
        """

        # 1. Get current time
        now = time.time()
        
        # 2. If now - last_time >= interval → allow frame
        if now - self.last_time >= self.interval:
            self.last_time = now
            return True
            
        # 3. Else → return False
        return False