FROM python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIPER_REVISION=639388b6317fc4731e91d53da42aea68fd4166ff

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIPER_REVISION=${PIPER_REVISION} \
    PYTHONUNBUFFERED=1 \
    PIPER_DATA_DIR=/data \
    GRADIO_TEMP_DIR=/data/tmp \
    HOME=/tmp \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ffmpeg espeak-ng libsndfile1 libgomp1 ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt piper-constraints.txt /app/
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.6.0 --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

RUN git clone https://github.com/OHF-Voice/piper1-gpl.git /opt/piper1-gpl \
    && cd /opt/piper1-gpl \
    && git checkout "${PIPER_REVISION}" \
    && python -m pip install --no-cache-dir --constraint /app/piper-constraints.txt -e '.[train]' \
    && python -m pip install --no-cache-dir 'scikit-build<1' 'cmake>=3.26,<4' \
    && python setup.py build_ext --inplace \
    && ./build_monotonic_align.sh

COPY app /app/app
COPY prompts /app/prompts
COPY README.md THIRD_PARTY_NOTICES.md /app/

RUN useradd --uid 10001 --create-home trainer \
    && mkdir -p /data \
    && chown -R trainer:trainer /app /data
ENV NUMBA_CACHE_DIR=/tmp/numba-cache
USER trainer

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3)" || exit 1

CMD ["python", "-m", "app.main"]
