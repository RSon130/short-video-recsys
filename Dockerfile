FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# faiss-gpu (PyPI) is deprecated for CUDA 12 — use the cu12 package instead.
# torch is already provided by the base image; do not reinstall from pip.
RUN pip install --no-cache-dir faiss-gpu-cu12>=1.8.0 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
