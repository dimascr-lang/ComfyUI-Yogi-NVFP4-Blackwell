import torch
from safetensors.torch import load_file, save_file

print("--- Starting Column-wise Quantization for NextDiT Layouts ---")

input_path = "C:/ComfyUI_windows_portable/ComfyUI/models/checkpoints/zimageTurboByStable_2603Fp8.safetensors"
output_path = "C:/ComfyUI_windows_portable/ComfyUI/models/checkpoints/zimageTurboByStable_2603_NVFP4_NoEncoder.safetensors"

weights = load_file(input_path)
quantized_weights = {}

for key, tensor in weights.items():
    if "pad_token" in key or "bias" in key or "norm" in key or "embedder" in key or tensor.ndim < 2:
        quantized_weights[key] = tensor.to(torch.float16)
    else:
        working_tensor = tensor.to(torch.float32)
        
        # Split packed QKV layers into 3 individual heads to protect scale boundaries
        if working_tensor.size(0) == 11520:
            chunks = torch.chunk(working_tensor, 3, dim=0)
            scales_list = []
            q_chunks = []
            for chunk in chunks:
                max_vals = chunk.abs().max(dim=0, keepdim=True)
                scale_chunk = max_vals / 7.0
                scale_chunk[scale_chunk == 0] = 1.0
                q_chunk = (chunk / scale_chunk).round().clamp(-8, 7).to(torch.int8)
                scales_list.append(scale_chunk.to(torch.float16))
                q_chunks.append(q_chunk)
            quantized_weights[key] = torch.cat(q_chunks, dim=0)
            quantized_weights[f"{key}.scale"] = torch.cat(scales_list, dim=0)
        else:
            # Standard structural layers processed down to int8 boundaries
            max_vals = working_tensor.abs().max(dim=0, keepdim=True)
            scale = max_vals / 7.0
            scale[scale == 0] = 1.0
            q_tensor = (working_tensor / scale).round().clamp(-8, 7).to(torch.int8)
            quantized_weights[key] = q_tensor
            quantized_weights[f"{key}.scale"] = scale.to(torch.float16)

save_file(quantized_weights, output_path)
print("--- Quantization Complete: Output compiled under dimension layouts ---")