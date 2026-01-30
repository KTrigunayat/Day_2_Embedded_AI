# Day 2 Embedded AI: Edge Camera Pipeline

This repository contains an optimized camera pipeline designed for Edge AI devices like the **NVIDIA Jetson Nano**. The pipeline handles video capture, preprocessing, simulated inference, and performance monitoring with strict resource constraints.

## 🚀 Project Overview

Running AI on edge devices requires careful resource management. This project implements a full pipeline that prioritizes:
1.  **Low Latency**: Minimizing delay between capture and inference.
2.  **Memory Efficiency**: Preventing memory leaks and excessive allocations.
3.  **Throughput Control**: Managing FPS to avoid overheating or bottlenecking.

## 📂 Project Structure

```text
edge_camera_pipeline/
├── camera/
│   └── webcam.py       # wrapper for cv2.VideoCapture with instant resolution setting
├── inference/
│   └── dummy_model.py  # Simulates model latency (~30ms) for testing flow
├── pipeline/
│   ├── preprocess.py   # Resizing and in-place normalization (crucial for Edge)
│   └── sampler.py      # Logic to control FPS and drop excess frames
├── utils/
│   └── metrics.py      # Real-time FPS and RAM usage monitoring
└── main.py             # Orchestrates the entire pipeline
```

## 🛠️ Optimizations for Edge AI (Jetson Nano)

The code follows strictly defined rules for edge optimization:

1.  **Never Load Full Dataset**: Images are processed uniquely as they arrive from the stream.
2.  **Resize Early**: Frames are resized immediately in `preprocess.py` before any other operation to reduce memory bandwidth.
3.  **Normalize Once**: Normalization is performed in-place (`/= 255.0`) on the float32 array to avoid creating temporary copies of large image matrices.
4.  **Use Generators / Samplers**: The `FrameSampler` ensures we only process what we can handle (e.g., 5 FPS target), simply dropping other frames to save CPU cycles.
5.  **Measure Memory**: Built-in monitoring allows real-time tracking of RAM usage to detect leaks early.

## 📦 Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install opencv-python psutil numpy
```

## 🏃‍♂️ How to Run

Navigate to the project root and run the main pipeline script:

```bash
python edge_camera_pipeline/main.py
```

You should see output similar to:

```text
FPS: 5.02 | Memory: 24.50 MB
FPS: 4.98 | Memory: 24.51 MB
...
```

Press `Ctrl+C` to stop the pipeline safely.