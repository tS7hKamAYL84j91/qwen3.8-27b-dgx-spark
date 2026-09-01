# Training container for the admin/Q post-training program (spec 0001).
# NVIDIA official DGX Spark path: NGC PyTorch 25.11 (CUDA 13.0) + Unsloth/TRL.
# Build:  docker build -f training/Dockerfile -t qwen38-train training/
FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN pip install --no-cache-dir \
      transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1" && \
    pip install --no-deps unsloth unsloth_zoo bitsandbytes

# GB10 / sm_121a notes: if Triton kernel compilation fails at runtime, fall
# back to the unslothai Dockerfile_DGX_Spark recipe (source-built triton
# c5d671f + xformers TORCH_CUDA_ARCH_LIST=12.1) per the NVIDIA playbook
# troubleshooting section.

WORKDIR /workspace
CMD ["/bin/bash"]