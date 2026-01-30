
import torch
print(f"Supported Engines: {torch.backends.quantized.supported_engines}")
try:
    print("Testing get_default_qconfig('onednn')...")
    qconfig = torch.quantization.get_default_qconfig('onednn')
    print("Success for onednn")
except Exception as e:
    print(f"Failed for onednn: {e}")

try:
    print("Testing get_default_qconfig('fbgemm')...")
    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print("Success for fbgemm")
except Exception as e:
    print(f"Failed for fbgemm: {e}")
