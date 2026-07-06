FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/SeedDataGen

COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /usr/local/bin/uv

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

CMD ["sleep", "infinity"]