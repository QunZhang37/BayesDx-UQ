# Minimal CUDA-ready image for training (adjust base as needed)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -r requirements.txt
CMD ["/bin/bash"]
