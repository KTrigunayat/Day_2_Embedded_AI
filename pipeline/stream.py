# stream.py
# Goal: Create a lazy image stream using Python generators

from pipeline.loader import read_image
from pipeline.preprocess import preprocess_image


import threading
import queue

def image_stream(image_paths, queue_size=4):
    """
    Given a list of image file paths, stream preprocessed images
    one-by-one using a generator.
    
    Optimization:
    - Uses a background thread to read and process images (Prefetching)
    - Decouples Disk I/O from Main Thread usage
    """
    
    # FIFO Queue
    q = queue.Queue(maxsize=queue_size)
    
    # -------------------------------------------------------------
    # Producer Thread: Reads disk -> Preprocess -> Puts in Queue
    # -------------------------------------------------------------
    def producer():
        for path in image_paths:
            img = read_image(path)
            
            # Skip corrupted images
            if img is None:
                continue
                
            # CPU-bound preprocessing (happens in parallel with main loop's inference)
            processed_img = preprocess_image(img)
            
            # Blocks if queue is full (Backpressure)
            q.put(processed_img)
        
        # Signal 'Done'
        q.put(None)

    # Start the thread
    t = threading.Thread(target=producer, daemon=True)
    t.start()

    # -------------------------------------------------------------
    # Consumer (Main generator): Yields to the main loop
    # -------------------------------------------------------------
    while True:
        item = q.get()
        if item is None:
            break
        yield item