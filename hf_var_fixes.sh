#!/usr/bin/env sh

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
CA="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$CA"
export SSL_CERT_FILE="$CA"
export CURL_CA_BUNDLE="$CA"
