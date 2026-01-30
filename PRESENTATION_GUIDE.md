# Embedded AI Presentation Guide

This document provides a comprehensive guide for presenting the **Embedded AI Day 2 Project**, covering assignment questions, MobileNet architecture, quantization strategies, and live demonstration steps.

---

## 1. Assignment & Modification Questions representing "why this design?"
*Based on the implementation choices and performance results observed in the project.*

### Q1: Why do we care about memory on Jetson?
**Context**: The Jetson Nano/Orin Nano has shared memory (CPU+GPU).
**Answer**: 
- **Resource Constraints**: Edge devices like Jetson Orin Nano typically have 4GB-8GB of unified RAM. This must support the OS, background tasks, the Python runtime, and the AI model/data.
- **Risk**: Exceeding the standard threshold (e.g., >3.5GB on a 4GB board) leads to swapping (thrashing), extreme latency spikes, or OOM (Out of Memory) kills, causing the application to crash.
- **Solution in Code**: We use `psutil` to monitor memory in real-time (`utils/monitor.py`) and optimize data loading to never hold the full dataset in RAM.

### Q2: Why pre-resize images before normalization (or boolean logic)?
**Context**: `pipeline/preprocess.py` executes `cv2.resize` *before* converting to `float32`.
**Answer**:
- **Speed & Memory**: An input image might be 4K or 1080p. Converting a 1080p image (1920x1080) to float32 creates a ~24MB array. Resizing it first to 224x224 (MobileNet input) creates a tiny 0.6MB array.
- **Efficiency**: All subsequent math operations (normalization, transposes) are 20x-100x faster because they operate on significantly fewer pixels.

### Q3: Why use Float32 instead of Float64 (Python default)?
**Context**: `img.astype(np.float32)` is explicitly used.
**Answer**:
- **Hardware Acceleration**: Neural Processing Units (NPUs) and GPUs are optimized for FP32 (or FP16/INT8). They often cannot process FP64 natively, requiring expensive casting drivers.
- **Memory Bandwidth**: FP64 takes 2x the memory bandwidth of FP32. On shared-memory architectures like Jetson, bandwidth is a precious resource.

### Q4: Comparison Results (Float32 vs Float64, Image Sizes)
*Based on screenshots in `results_/`:*
- **Observation**: Using Float64 consistently results in lower FPS and higher memory usage compared to Float32.
- **Observation**: Increasing batch size or image resolution (e.g., 512x512 vs 224x224) drastically increases latency, often non-linearly, due to cache misses and bandwidth saturation.

---

## 2. MobileNet Architecture Explanation

### Overview
**MobileNetV2** is a convolutional neural network architecture specifically designed for mobile and resource-constrained environments. It improves upon MobileNetV1 by introducing **Inverted Residuals** and **Linear Bottlenecks**.

### Key Components

#### 1. Depthwise Separable Convolutions
Standard convolutions perform spatial filtering and channel mixing simultaneously, which is computationally expensive. MobileNet splits this into:
- **Depthwise Conv**: Filters each input channel independently (Spatial filtering).
- **Pointwise Conv (1x1)**: Combines the outputs of the depthwise conv (Channel mixing).
**Benefit**: Reduces computation (FLOPs) and parameters by ~8-9x compared to standard convolutions with minimal accuracy loss.

#### 2. Inverted Residuals
- **Traditional ResNet**: Wide (High dim) → Narrow (Bottleneck) → Wide (High dim).
- **MobileNetV2**: Narrow (Low dim) → Wide (Expansion) → Narrow (Bottleneck).
- **Why?**: The network expands low-dimensional compressions to high dimensions to filter features in a rich space, then projects them back down to a compact representation. This is more memory-efficient because the "skip connections" link the small tensors (bottlenecks), keeping memory footprint low during inference.

#### 3. Linear Bottlenecks
The last convolution in a block (the projection back to "narrow") uses a **Linear Activation** (no ReLU).
- **Reason**: ReLU destroys information in low-dimensional spaces (zeroing out negatives). By strictly using linear layers for the low-dimensional bottlenecks, the model preserves critical feature information.

---

## 3. Quantization Strategies
Quantization reduces model size and inference latency by mapping high-precision values (Float32) to lower precision (Int8).

### Method A: Dynamic Quantization
- **How it works**: Weights are quantized to Int8 ahead of time. Activations are read as Float32 and quantized *dynamically* on the fly just before operations throughout the network.
- **Use Case**: Best for LSTMs, RNNs, or Transformer models (BERT) where compute is dominated by memory bandwidth of loading weights.
- **Limitations**: Less effective for CNNs (like MobileNet) because CNNs are compute-bound by convolutions, and the overhead of on-the-fly quantization of activations can negate benefits.

### Method B: Static Quantization (Post-Training Quantization - PTQ) <span style="color:green; font-weight:bold">[Recommended for Jetson]</span>
- **How it works**: Both Weights AND Activations are quantized to Int8.
- **Calibration**: Requires a "calibration step" where we run representative data through the model to determine the range (scale and zero-point) of activations at each layer.
- **Benefit**: Matrix multiplications become Int8 instructions (faster on CPU/DSP) and memory bandwidth is halved effectively.
- **Jetson Specifics**: We use the `qnnpack` backend in PyTorch, which is optimized for ARM CPUs found on Jetson boards.

---

## 4. Live Demonstration Plan (Group Usage)

### Setup (One time)
Ensure dependencies are installed on the Jetson Board:
```bash
pip install -r edge_mobilenent_pipeline/requirements.txt
```

### Demo 1: The Optimized Pipeline (Float32)
Show the real-time performance of the camera pipeline.
1. **Open Terminal**.
2. **Run**:
   ```bash
   python edge_mobilenent_pipeline/main.py
   ```
3. **Show**:
   - Point the webcam at objects.
   - Observe the **FPS** and **Memory Usage** in the terminal output.
   - Note how it stays stable (e.g., ~25MB memory). (Show `utils/monitor.py` code).

### Demo 2: Quantization in Action
Demonstrate the compression and speedup of quantization.
1. **Open Terminal**.
2. **Run the quantization script**:
   ```bash
   python edge_mobilenent_pipeline/quantize.py
   ```
3. **Explain Output**:
   - **Float32 Model Size**: ~9-14 MB.
   - **Int8 Model Size**: ~2.5-3.5 MB (4x reduction).
   - **Inference Latency**: Compare the 'ms' per inference. Static Int8 should be faster than Float32 on the CPU.
4. **Conclusion**: Quantization enables running larger models on smaller chips by trading a tiny bit of accuracy for significant speed and memory gains.

### Supportive Code Base
- **`edge_camera_pipeline/`**: The pure camera handling and resource management code.
- **`edge_mobilenent_pipeline/inference/mobilenet.py`**: The model definition.
- **`edge_mobilenent_pipeline/quantize.py`**: The script generating static/dynamic quantized models.
