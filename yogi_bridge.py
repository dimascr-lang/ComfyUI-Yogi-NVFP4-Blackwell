import os
import torch
from server import PromptServer
from aiohttp import web

_STORAGE = {"c": None, "p": None}
_KEY_HASH = os.environ.get("YOGI_CORE_KEY", "UNAUTHORIZED_BASELINE_SHUTDOWN")

@PromptServer.instance.routes.post("/yogi/inject_gemma")
async def _inbound_handler(request):
    try:
        if request.headers.get("X-Yogi-Auth") != _KEY_HASH:
            return web.Response(text="Secured Layer Locked. Authorization token required.", status=403)
            
        payload = await request.json()
        c_layer = payload.get("conditioning")
        p_layer = payload.get("pooled")
        
        if c_layer:
            _STORAGE["c"] = torch.tensor(c_layer, dtype=torch.bfloat16, device="cuda").contiguous()
        if p_layer:
            _STORAGE["p"] = torch.tensor(p_layer, dtype=torch.bfloat16, device="cuda").contiguous()
            
        return web.Response(text="SYNC_OK", status=200)
    except Exception:
        return web.Response(text="ERR_STREAM", status=500)

class YogiGemmaBridgeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"token_length": ("INT", {"default": 256, "min": 64, "max": 512, "step": 64})}}
    
    RETURN_TYPES = ("CONDITIONING", "POOLED_OUTPUT")
    RETURN_NAMES = ("cond", "pooled")
    FUNCTION = "_resolve"
    CATEGORY = "YogiTools"

    def _resolve(self, token_length):
        if _STORAGE["c"] is None:
            return (torch.zeros((1, token_length, 4096), dtype=torch.bfloat16, device="cuda"),
                    torch.zeros((1, 4096), dtype=torch.bfloat16, device="cuda"))
        return (_STORAGE["c"], _STORAGE["p"])

NODE_CLASS_MAPPINGS = {"YogiGemmaBridge": YogiGemmaBridgeNode}
