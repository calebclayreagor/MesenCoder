#!/bin/bash

set -euo pipefail

# usage check
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <url> [id/geo_accession]"
  exit 1
fi

URL="$1"
FN_ENCODED="${URL##*/}"
FN_DECODED=$(printf '%b' "${FN_ENCODED//%/\\x}")
FN_DECODED=$(echo "$FN_DECODED" | cut -d'?' -f1)
BASENAME="${FN_DECODED%.gz}"
BASENAME="${BASENAME%.tar}"

# determine ID/GEO accession
if [[ $# -eq 2 ]]; then
  ID="$2"
else
  ID=$(echo "$FN_DECODED" | cut -d'_' -f1 | cut -d'.' -f1)
fi

TIME=$(date +"%Y%m%d_%H%M%S")
FN_LOG="logs/${ID}_${TIME}.log"

# redirect stdout and stderr to logfile
exec > >(tee -i "$FN_LOG") 2>&1

echo "==== Data Download and Extraction ===="
echo "Started at: $(date)"
echo "URL: $URL"
echo "Data ID or GEO Accession: $ID"
echo "Log file: $FN_LOG"
echo "======================================"

RAW_DIR="raw/$ID"
UNZIP_DIR="unzip/$ID"
RAW_PATH="$RAW_DIR/$FN_DECODED"
UNZIP_PATH="$UNZIP_DIR/$BASENAME"

mkdir -p "$RAW_DIR"
mkdir -p "$UNZIP_DIR"

echo "Downloading to: $RAW_PATH"
curl -L "$URL" -o "$RAW_PATH"

if [[ ! -s "$RAW_PATH" ]]; then
  echo "Download failed or file is empty: $RAW_PATH"
  exit 1
fi

# extract based on file type
if [[ "$FN_DECODED" == *.tar ]]; then
  echo "Extracting tar archive to: $UNZIP_DIR"
  tar -xf "$RAW_PATH" -C "$UNZIP_DIR"
elif [[ "$FN_DECODED" == *.gz ]]; then
  echo "Unzipping to: $UNZIP_PATH"
  gunzip -c "$RAW_PATH" > "$UNZIP_PATH"
elif [[ "$FN_DECODED" == *.h5ad || "$FN_DECODED" == *.csv ]]; then
  echo "Raw files copied to: $UNZIP_PATH"
  cp "$RAW_PATH" "$UNZIP_PATH"
else
  echo "Unsupported file type: $FN_DECODED"
  exit 1
fi

echo "Done"
echo "Raw file saved to:     $RAW_PATH"
echo "Extracted file(s) saved to: $UNZIP_DIR"
echo "Finished at: $(date)"
