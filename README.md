\# Yogi NVFP4/INT4 Blackwell Inference \& Optimization Toolkit



This repository contains a low-level backend integration node and diagnostic tooling designed to run NextDiT/Lumina2 architectures (specifically tested on ZIT 2603 Pro) natively on NVIDIA Blackwell (RTX 50-series) hardware under Windows ComfyUI Portable setups.



\## Repository Structure



\- `custom\_nodes/ComfyUI-Yogi-NVFP4/\_\_init\_\_.py` -> Custom model loader implementing memory-efficient mmap loading and low-level `torch.\_scaled\_mm` patching.

\- `convert\_with\_compressor.py` -> Axis-aligned structural PTQ quantization script featuring tripartite QKV split-axis scaling.

\- `compare\_signals.py` -> Diagnostic matrix validator evaluating mean squared error (MSE) between original FP8 checkpoints and quantized states.



\---



\## Step-by-Step Execution Guide



\### Step 1: Quantization \& Structural Axis Alignment

Run the compressor script to compress the target FP8 weights down to 4-bit elements. The script automatically handles the NextDiT packed QKV attention layout by splitting the 11520 matrix channels into 3 independent 3840 headers to align the scale broadcasting arrays:



```bash

.\\python\_embeded\\python.exe convert\_with\_compressor.py

```



\### Step 2: Signal Validation \& MSE Diagnostics

Run the matrix signal sniffer to cross-evaluate the hidden states recovery vector. Under correct column-wise axis broadcasting, the reconstruction baseline yields an exact signal precision error metric of `MSE = 0.000769`:



```bash

.\\python\_embeded\\python.exe compare\_signals.py

```



\### Step 3: Deployment in ComfyUI (Blackwell Native Loop)

1\. Move the `ComfyUI-Yogi-NVFP4` folder directly into your `ComfyUI/custom\_nodes/` directory.

2\. Launch ComfyUI via your standard NVIDIA GPU bat file.

3\. Insert the `🧘‍♂️ Yogi NVFP4 Model Loader` node into your workflow. 



\### Current Execution Matrix \& Staging Limits:

\- \*\*Low-Memory Path (mmap):\*\* The backend leverages `safetensors.torch.safe\_open` to stream model keys directly to VRAM, ensuring 0 MB allocation spikes in system RAM during model initialization.

\- \*\*Hardware Layer:\*\* Validated layers are fed strictly to Blackwell’s 5th-gen Tensor Cores using `torch.\_scaled\_mm` with real-time activation scaling (`scale\_a`).

\- \*\*Current Barrier:\*\* Standard MinMax per-channel PTQ destroys extreme activation outliers, causing high-frequency grid noise pattern anomalies on pure INT4. To bypass this, an internal dynamic fallback is configured, which currently stages a temporary bfloat16 reconstruction footprint. 

\- \*\*Next Step:\*\* To unlock full, noise-free execution natively under `torch.\_scaled\_mm` at pure INT4 boundaries, the weights must be calibrated via an SVDQuant-style data routine (isolating the 1% outlier heads to native FP16 and leaving 99% in INT4).



\---

Developed by Dima Skryabin (GMT+3)

