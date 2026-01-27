# specific_implementation_details
# Edge AI Implementation Layout & Design Choices

This document details the architectural decisions and answers the specific questions embedded in the codebase comments. These choices are critical for deploying AI pipelines on resource-constrained Edge AI devices (e.g., NVIDIA Jetson, Raspberry Pi).

## 1. System Monitoring (`utils/monitor.py`)

### **Code Implementation**
- **Memory Tracking**: Uses `psutil.virtual_memory()` to get the current system RAM usage in MB.
- **FPS Counter**: Uses a cumulative timer (`elapsed = current - start`) rather than a simple per-second reset interval for more stable long-term averaging.

### **Q&A from Docstrings**
**Q: Why do we care about memory on Jetson?**
*   **A**: Edge devices like the Jetson Nano or Orin Nano typically have limited shared memory, often just 4GB or 8GB shared between CPU and GPU. Hence, we need to optimize the memory usage on edge AI devices to be able to utilize the memory perfectly. Hence, we also check regularly the amount of memory in use in order to prevent over heating and other isssues that might arise because of the excessive memory usage.
**Q: What happens if we cross 3.5 GB RAM?**
*   **A**: The Jetson Orin Nano board that we have has around 4 GB of RAM, out of which some memory will be used for running the basic OS programs, if we exceed the memory usage above 3.5 GB, it can lead to the issues in performance and latency and it might have to shutdown some background tasks to run the application, making it difficuilt for the jetson board to function.

**Q: Why use elapsed time instead of counting per second?**
*   **A**: Counting per second increases the need for running an extra process continuously and storage of data values at every time stamp is again an issue, hence, we prefer batch processing on edge AI devices and hence when we calculate the elapsed time, we are not storing data every second, instead in batches.

---

## 2. Data Loading (`pipeline/loader.py`)

### **Code Implementation**
- **Lazy Discovery**: `load_image_paths` only collects file strings, not pixel data.
- **On-Demand Reading**: `read_image` loads a singe file from disk using OpenCV only when correctly requested.

### **Q&A from Docstrings**
**Q: Why is this better for edge devices? (Returning paths instead of images)**
*   **A**: Loading a dataset of even moderate size would instantly consume a quarter of the device's RAM. Storing strings (paths) takes negligible memory, allowing us to scale to millions of images without crashing.

**Q: Why should reading be separate from preprocessing?**
*   **A**:
    1.  **Modularity**: We might change how we read files (e.g., from a camera feed or network stream) without changing how we resize/normalize them.
    2.  **Error Isolation**: If a file read fails (corrupted disk), we can catch it before it enters the math-heavy processing logic.

---

## 3. Preprocessing (`pipeline/preprocess.py`)

### **Code Implementation**
- **Early Resizing**: `cv2.resize` is called *first*.
- **Float32 Selection**: Explicit `.astype(np.float32)`.
- **In-Place Normalization**: `img /= 255.0` modifies the array in memory without creating a copy.

### **Q&A from Docstrings**
**Q: Why resize first?**
*   **A**: Speed and Memory.
    *   If an input image is 4K (3840x2160) and the model needs 224x224, converting the 4K image to float32 first would create a massive array (~25MB+).
    *   Resizing first reduces it to a tiny array (~0.15MB) immediately. Subsequent operations (normalization, conversion) are then 100x faster because they operate on fewer pixels.

**Q: Why not normalize uint8 directly?**
*   **A**: `uint8` handles integers 0-255. Neural networks typically expect inputs in range [0, 1] or [-1, 1] with high precision. Integer division on uint8 results in loss of precision (0 or 1 only), destroying the image information.

**Q: Why float32 instead of default NumPy float?**
*   **A**:
    *   **Memory**: `float64` (default python float) takes double the RAM of `float32`.
    *   **Hardware Compatibility**: GPUs and NPU accelerators are optimized for `float32` (or `float16`). Passing `float64` often forces the driver to cast it back down anyway, wasting CPU cycles and bus bandwidth.

---

## 4. Streaming (`pipeline/stream.py`)

### **Code Implementation**
- **Generator Pattern**: Uses the `yield` keyword to produce one processed frame at a time.
- **Error Handling**: Checks `if img is None` and continues, preventing pipeline crashes on bad files.

### **Q&A from Docstrings**
**Q: Constraints: Do NOT return a list. Do NOT load all images at once.**
*   **A**: Returning a list `[img1, img2, ...]` forces all processed images to stay in RAM. A generator only holds **one** image in RAM at any given instant.

**Q: What happens if image is corrupted?**
*   **A**: In a production loop, a single corrupted file (which returns `None` from `read_image`) would throw an exception if not handled. The logic `if img is None: continue` ensures the robot/camera keeps running even if one frame is bad.

**Q: What if we want to log progress later?**
*   **A**: Because it is a generator, we can wrap it easily with progress libraries (like `tqdm`) or inject logging in the loop without rewriting the core data loading logic.
