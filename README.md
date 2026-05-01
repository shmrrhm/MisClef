# 🎵 MisClef
MisClef turns the "mischief" of complex sheet music into readable data. Designed for the musically illiterate — whether you're struggling with the staff or feeling "clef-less," MisClef transcribes chaos into clarity. 🎹

## How it works 👁️

MisClef uses **computer vision** and **Optical Music Recognition (OMR)** — it analyses sheet music as an image, not as structured data. The pipeline renders each PDF page to a pixel image, detects staff lines geometrically, and then uses a deep-learning UNet model to locate note heads directly in the image. Because it reads pixels rather than file metadata, it works on **any PDF** — including scanned or photographed scores — with no requirement for MusicXML, MIDI, or any other structured music notation format.

## Credits 🙏

Notehead detection is powered by **[oemer](https://github.com/BreezeWhite/oemer)** by BreezeWhite — an end-to-end optical music recognition library whose UNet segmentation model is used here to accurately locate note heads on each staff.

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Performance 🚀

By default, MisClef runs inference on CPU. For significantly faster processing, install the GPU-accelerated ONNX Runtime along with CUDA and cuDNN:

1. **Install CUDA** — Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (check the ONNX Runtime release notes for the supported version).
2. **Install cuDNN** — Download [cuDNN](https://developer.nvidia.com/cudnn) matching your CUDA version and follow NVIDIA's installation guide.
3. **Install ONNX Runtime with GPU support** — Replace the CPU-only package with the GPU build:

   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```

When a compatible GPU is detected, inference will automatically use CUDA, dramatically reducing processing time for multi-page scores.

### Benchmarks

Measured on a several score sheets (oemer UNet, CUDA execution provider):

| Hardware | CUDA | Score | Pages | Min | Avg | Max |
|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GB) | 12.6 | Secrets | 7 | 22 s/page | 22 s/page | 24 s/page |
| NVIDIA GeForce RTX 3070 (8 GB) | 12.6 | Nocturne | 4 | 25 s/page | 25 s/page | 25 s/page |