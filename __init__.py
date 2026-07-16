import torch
import safetensors.torch
import comfy.model_management
import comfy.sd
import folder_paths

class YogiNVFP4Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("checkpoints"), ),
        }}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_yogi_model"
    CATEGORY = "YogiTools"

    def load_yogi_model(self, model_name):
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        print(f"[YogiLoader] Initializing Hybrid Blackwell Inference: {model_path}")
        
        # 1. Zero-RAM Stage using mmap mapping
        handle = safetensors.torch.safe_open(model_path, framework="pt", device="cpu")
        all_keys = handle.keys()
        
        scales = {}
        for k in all_keys:
            if k.endswith(".scale"):
                scales[k] = handle.get_tensor(k).cuda().to(torch.float16).view(-1)
        
        native_nvfp4_dict = {}
        for key in all_keys:
            if key.endswith(".scale"):
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype == torch.int8:
                native_nvfp4_dict[key] = tensor.cuda()
            else:
                native_nvfp4_dict[key] = tensor.cuda().to(torch.float16)
        
        model_options = {}
        model = comfy.sd.load_diffusion_model_state_dict(native_nvfp4_dict, model_options=model_options)
        
        # 2. Low-Level Patched Linear Layer for KSampler Execution Loop
        def native_blackwell_nvfp4_linear(w, x, bias=None):
            weight_name = getattr(w, "comfy_name", None)
            
            if weight_name and f"{weight_name}.scale" in scales:
                s_flat = scales[f"{weight_name}.scale"]
                
                if s_flat.size(0) != w.size(0):
                    s_flat = s_flat[:w.size(0)]
                
                s_flat = s_flat.clone()
                s_flat[s_flat == 0] = 1.0
                
                # Dynamic shape protection for irregular tensor dimensions
                if w.size(0) % 16 != 0 or w.size(1) % 16 != 0 or x.size(-1) % 16 != 0:
                    q_weight = w.to(torch.bfloat16) * s_flat.view(-1, 1).to(torch.bfloat16)
                    return torch.nn.functional.linear(x, q_weight.to(x.dtype), bias)
                
                # Active torch._scaled_mm execution path for compatible hardware layouts
                x_fp16 = x.to(torch.float16)
                out = torch._scaled_mm(x_fp16, w.t(), scale_a=torch.tensor([1.0], device=x.device), scale_b=s_flat)
                
                if bias is not None:
                    out += bias.to(out.dtype)
                return out.to(x.dtype)
            
            return torch.nn.functional.linear(x, w.to(x.dtype), bias)

        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}
        model_options["transformer_options"]["patches"] = {"linear": native_blackwell_nvfp4_linear}
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return (model,)

NODE_CLASS_MAPPINGS = {"YogiNVFP4Loader": YogiNVFP4Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"YogiNVFP4Loader": "🧘‍♂️ Yogi NVFP4 Model Loader"}