# Kokoro local CUDA on Windows

Profile validated on Dell Precision 5550: i7-10875H, 32 GB RAM,
Quadro T2000 Max-Q 4 GB, NVIDIA driver 610.60.

## Run

From the repository root in PowerShell:

```powershell
.\run-performance.ps1 -OpenBrowser
```

Open http://127.0.0.1:7860. Select **NVIDIA CUDA**, enter English text,
and click **Generate**. Use **Stream** for all chunks of long text.
The demo provides American and British English voices; Vietnamese is not supported.
Stop a foreground session with Ctrl+C. Set another port with `-Port 7861`.

## Runtime profile

- Python 3.12, PyTorch 2.8.0+cu126, CUDA runtime 12.6, FP32.
- CUDA inference with `torch.inference_mode()`, cuDNN enabled and benchmark enabled.
- 2 CPU intra-op threads, 1 inter-op thread; Windows process priority AboveNormal.
- Model and default voice stay loaded; GPU warms up before serving.
- Inference events share one queue, with one active request to avoid competing for VRAM.
- CPU model loads only if explicitly selected. Launch fails if CUDA is unavailable.
- eSpeak NG is bundled in `.venv`; a system MSI install is unnecessary.

Benchmark on 2026-10-27 compared 1/2/4/8/16 CPU threads, cuDNN benchmark on/off,
with 3 timed passes over two English texts after warm-up. The fastest measured
median was **0.376 seconds/request** with 2 threads and cuDNN benchmark enabled.
A sample produced **11.1 seconds of audio in 0.526 seconds** (about 21x real time),
with **530 MiB peak PyTorch allocated VRAM**. Driver/context memory is additional.
These are model inference timings; UI, text processing and file encoding add latency.
The installed Windows power plan already permits 100% maximum CPU on AC power.

## Install from scratch

Requires `uv` and an NVIDIA driver compatible with CUDA 12.6. A separate CUDA Toolkit
installation is unnecessary for the prebuilt PyTorch wheel.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --link-mode copy
uv pip install --python .venv/Scripts/python.exe -r demo/requirements-local.txt --link-mode copy
```

The first run downloads Kokoro model weights, the default voice and the spaCy English
model. Runtime logs and benchmark artifacts are kept in the ignored `.local` directory.

For a background session started during setup, find its Python PID with:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 7860 | Select-Object OwningProcess
# Stop-Process -Id <OwningProcess>
```
