# mimicry-auth-demo — Facial Expression Biometric Auth Demo

Streamlit application demonstrating facial expression biometric authentication using GOST R 52633.5-2011 NPBK.

## Setup (local)

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install local library dependencies
pip install -e ../mimicry-preproc
pip install -e ../npbk
pip install -e ".[dev]"
```

## Generate SSL certificate (required for webcam / WebRTC)

WebRTC requires HTTPS. Generate a self-signed cert:

```bash
bash scripts/gen_ssl.sh
```

## Run

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
# Open: https://192.168.0.25:8501
# (Browser will warn about self-signed cert → click Advanced → Proceed)
```

## Run with Docker

```bash
docker-compose up
# Open: https://localhost:8501
```

## Pages

| Page | Description |
|---|---|
| Home | Overview and enrolled profiles |
| Enrollment | Record facial expressions → train NPBK → save profile |
| Authentication | Verify identity against enrolled profile |
| Pipeline Inspector | Live per-stage visualization + parameter tuning |
| Dataset Browser | Browse RAVDESS/OULU-CASIA, preview recordings |
| Analysis | Stability charts, FAR/FRR metrics |
