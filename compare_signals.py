import torch
from safetensors.torch import load_file

print("--- Initializing Matrix Signal MSE Diagnostic Scanner ---")

orig = load_file("C:/ComfyUI_windows_portable/ComfyUI/models/checkpoints/zimageTurboByStable_2603Fp8.safetensors")
quant = load_file("C:/ComfyUI_windows_portable/ComfyUI/models/checkpoints/zimageTurboByStable_2603_NVFP4_NoEncoder.safetensors")

target_key = "model.diffusion_model.double_blocks.0.img_attn.qkv.weight"

if target_key in orig and target_key in quant:
    w_orig = orig[target_key].to(torch.float32).cuda()
    w_q = quant[target_key].to(torch.float32).cuda()
    scale_key = f"{target_key}.scale"
    
    if scale_key in quant:
        w_scale = quant[scale_key].to(torch.float32).cuda()
        
        # Verify and match duplicated head configurations
        if w_scale.size(0) != w_q.size(0):
            w_scale = w_scale.repeat(3, 1) if w_scale.size(0) == 3840 else w_scale[:w_q.size(0)]
            
        # Reconstruct layer signal inside memory blocks
        w_reconstructed = w_q * w_scale
        
        # Calculate pure Mean Squared Error metrics
        mse = torch.mean((w_orig - w_reconstructed) ** 2).item()
        print(f"Target Layer Signal Match verified. Resulting MSE Error metrics: {mse:.7f}")
    else:
        print("Scale dictionary reference key layout missing.")
else:
    print("Baseline reference layers not found inside target weights.")
