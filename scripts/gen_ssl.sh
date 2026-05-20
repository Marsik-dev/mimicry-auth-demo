#!/usr/bin/env bash
set -e
mkdir -p ssl
openssl req -x509 -newkey rsa:2048 \
    -keyout ssl/key.pem -out ssl/cert.pem \
    -days 3650 -nodes \
    -subj "/CN=localhost" \
    -addext "subjectAltName=IP:192.168.0.25,IP:127.0.0.1,DNS:localhost"
chmod 600 ssl/key.pem
echo "SSL certificate generated in ssl/"
echo "Open the app at: https://192.168.0.25:8501"
echo "(Browser will warn about self-signed cert — click 'Advanced > Proceed')"
