
import torch
import torch.quantization
from inference.mobilenet import MobileNetInference
import os
import copy
import time


class QuantizedModelWrapper(torch.nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModelWrapper, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def print_size_of_model(model, label="Model"):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    print(f"{label} Size: {size:.2f} MB")
    os.remove('temp.p')
    return size

def measure_inference_latency(model, device="cpu", input_shape=(1, 3, 224, 224), iterations=50):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
            
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    end_time = time.time()
    
    avg_pipeline = (end_time - start_time) / iterations * 1000 # ms
    print(f"Average Inference Latency ({device}): {avg_pipeline:.2f} ms")

def run_quantization():
    print("Loading MobileNetV2...")
    # Use CPU for quantization path
    wrapper = MobileNetInference(device="cpu")
    model = wrapper.model
    model.eval()

    print("\n[1] Baseline Float32 Model")
    print_size_of_model(model, "Float32")
    measure_inference_latency(model, "cpu")

    # --- Dynamic Quantization ---
    print("\n[2] Dynamic Quantization")
    print("Applying Dynamic Quantization (Weights only for Linear/RNN layers)...")
    # MobileNetV2 is mostly Conv layers, so Dynamic Quantization (which targets Linear) will have minimal effect on size/speed.
    # This is mainly for demonstration.
    quantized_dynamic = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    print_size_of_model(quantized_dynamic, "Dynamic Int8")
    # measure_inference_latency(quantized_dynamic, "cpu") # Might be slower on some backends if not optimized

    # --- Static Quantization (Post Training Quantization) ---
    print("\n[3] Static Quantization (PTQ)")
    print("Setting up QConfig for ARM (qnnpack)...")
    
    # Create a copy to avoid modifying original
    model_static = copy.deepcopy(model)
    # Wrap the model to include QuantStub and DeQuantStub
    model_static = QuantizedModelWrapper(model_static)
    
    # Determine the best available quantization backend
    supported_engines = torch.backends.quantized.supported_engines
    print(f"Supported engines: {supported_engines}")
    
    backend = 'none'
    qconfig_backend = 'fbgemm' # Default for x86 if unsure
    
    if 'qnnpack' in supported_engines:
        backend = 'qnnpack'
        qconfig_backend = 'qnnpack'
    elif 'fbgemm' in supported_engines:
        backend = 'fbgemm'
        qconfig_backend = 'fbgemm'
    elif 'onednn' in supported_engines:
        backend = 'onednn'
        qconfig_backend = 'fbgemm' # onednn usually works with fbgemm qconfig (symmetric)
    else:
        # Fallback based on machine if no optimized engine found/reported
        import platform
        machine = platform.machine().lower()
        if 'x86' in machine or 'amd64' in machine:
            qconfig_backend = 'fbgemm'
        else:
            qconfig_backend = 'qnnpack'
        print(f"Warning: No standard engine found in supported_engines. Defaulting qconfig to {qconfig_backend}.")

    print(f"Selected engine: {backend}")
    print(f"Selected qconfig: {qconfig_backend}")
    
    if backend != 'none':
        torch.backends.quantized.engine = backend
        
    model_static.qconfig = torch.quantization.get_default_qconfig(qconfig_backend)
    
    print("Preparing model...")
    torch.quantization.prepare(model_static, inplace=True)
    
    print("Calibrating with random data (Simulating 100 frames)...")
    # unique random inputs to simulate calibration
    with torch.no_grad():
        for _ in range(20):
            dummy_input = torch.randn(1, 3, 224, 224)
            model_static(dummy_input)
            
    print("Converting model to Int8...")
    torch.quantization.convert(model_static, inplace=True)
    
    print_size_of_model(model_static, "Static Int8")
    measure_inference_latency(model_static, "cpu")
    
    # Save the models
    torch.save(quantized_dynamic.state_dict(), "mobilenet_v2_dynamic.pth")
    torch.save(model_static.state_dict(), "mobilenet_v2_static.pth")
    print("\nModels saved: mobilenet_v2_dynamic.pth, mobilenet_v2_static.pth")

if __name__ == "__main__":
    run_quantization()
