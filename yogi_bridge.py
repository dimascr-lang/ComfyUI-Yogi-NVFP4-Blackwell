import torch
from server import PromptServer
from aiohttp import web

# Абсолютно открытый буфер без проверок хэш-ключей
_STORAGE = {"c": None, "p": None}

@PromptServer.instance.routes.post("/yogi/inject_gemma")
async def _inbound_handler(request):
    try:
        payload = await request.json()
        c_layer = payload.get("conditioning")
        p_layer = payload.get("pooled")
        
        # Нативный впрыск на транзисторы Blackwell с выравниванием памяти
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
        return {"required": {
            "token_length": ("INT", {"default": 256, "min": 64, "max": 512, "step": 64})
        }}
    
    RETURN_TYPES = ("CONDITIONING", "POOLED_OUTPUT")
    RETURN_NAMES = ("cond", "pooled")
    FUNCTION = "_resolve"
    CATEGORY = "YogiTools"

    def _resolve(self, token_length):
        # Железный предохранитель: если пульт еще ничего не прислал, выдаем чистый bfloat16-массив
        if _STORAGE["c"] is None or _STORAGE["p"] is None:
            c = torch.zeros((1, token_length, 4096), dtype=torch.bfloat16, device="cuda").contiguous()
            p = torch.zeros((1, 4096), dtype=torch.bfloat16, device="cuda").contiguous()
            return (c, p)
            
        return (_STORAGE["c"], _STORAGE["p"])

NODE_CLASS_MAPPINGS = {"YogiGemmaBridge": YogiGemmaBridgeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"YogiGemmaBridge": "🧘‍♂️ Yogi Gemma Text Stream Bridge"}

