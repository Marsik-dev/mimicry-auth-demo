FROM python:3.12-slim

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 openssl \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Generate self-signed SSL cert (WebRTC requires HTTPS)
RUN mkdir /ssl && \
    openssl req -x509 -newkey rsa:2048 \
        -keyout /ssl/key.pem -out /ssl/cert.pem \
        -days 3650 -nodes \
        -subj "/CN=mimicry-auth" \
        -addext "subjectAltName=IP:0.0.0.0,IP:127.0.0.1,DNS:localhost" && \
    chmod 600 /ssl/key.pem

WORKDIR /app

# Install library wheels from GitHub Releases
# These ARGs are set during docker build (or docker-compose)
ARG NPBK_WHL=npbk-0.1.0-py3-none-any.whl
ARG PREPROC_WHL=mimicry_preproc-0.1.0-py3-none-any.whl
ARG GH_NPBK_URL=https://github.com/Marsik-dev/npbk/releases/latest/download/${NPBK_WHL}
ARG GH_PREPROC_URL=https://github.com/Marsik-dev/mimicry-preproc/releases/latest/download/${PREPROC_WHL}

RUN pip install --no-cache-dir "${GH_NPBK_URL}" "${GH_PREPROC_URL}"

# Copy and install the demo app
COPY . .
RUN pip install --no-cache-dir -e "."

# Dataset is mounted at runtime: -v /path/to/datasets:/data/datasets
VOLUME /data/datasets
ENV DATASET_PATH=/data/datasets

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.sslCertFile=/ssl/cert.pem", \
     "--server.sslKeyFile=/ssl/key.pem"]
