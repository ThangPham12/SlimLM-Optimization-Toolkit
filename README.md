# 🚀 SlimLM-Optimization-Toolkit

[![Research](https://img.shields.io/badge/Focus-AI%20Optimization-blue.svg)](https://github.com/ThangPham12/SlimLM-Optimization-Toolkit)
[![Tech](https://img.shields.io/badge/Tech-PyTorch%20%2F%20TensorRT-orange.svg)](https://github.com/ThangPham12/SlimLM-Optimization-Toolkit)

A comprehensive toolkit designed for researchers and engineers to compress and optimize Large Language Models into efficient **Small Language Models (SLMs)** suitable for edge devices.

## ✨ Core Modules
- **Dynamic Pruning:** Layer-wise and head-wise pruning strategies for transformer architectures.
- **Quantization-Aware Training (QAT):** Support for INT8 and FP4 quantization schemes.
- **Knowledge Distillation:** Advanced teacher-student frameworks for maintaining performance during compression.

## 🚀 Quick Start
```python
from slimlm import Compressor
model = Compressor.load("llama-3-8b")
compressed_model = model.prune(sparsity=0.3).quantize(bits=4)
```
