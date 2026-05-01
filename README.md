# 🎵 MisClef
MisClef turns the "mischief" of complex sheet music into readable data. Designed for the musically illiterate — whether you're struggling with the staff or feeling "clef-less," MisClef transcribes chaos into clarity. 🎹

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