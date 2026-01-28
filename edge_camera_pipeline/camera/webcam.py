# webcam.py
# Responsibility:
# - Initialize a webcam
# - Set resolution
# - Capture one frame at a time
# - Release resources safely

import cv2


class Webcam:
    def __init__(self, cam_id=0, width=640, height=480):
        """
        Initialize webcam stream.

        Parameters:
        - cam_id: which camera (0 = default)
        - width/height: capture resolution

        Think:
        - Why control resolution early?
        - What happens if we skip .isOpened() check?
        """

        # 1. Create cv2.VideoCapture object
        self.cap = cv2.VideoCapture(cam_id)
        
        # 2. Set capture resolution (width + height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 3. Check if camera opened properly
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {cam_id}")

    def read(self):
        """
        Capture and return a single frame.

        Think:
        - What should you return if capture fails?
        - Why handle failure gracefully?
        """

        # 1. Read frame using cap.read()
        ret, frame = self.cap.read()
        
        # 2. Return frame if successful, else return None
        if not ret:
            return None
            
        return frame

    def release(self):
        """
        Release the webcam safely.
        Always call this when finished.
        """
        # Call cap.release()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()