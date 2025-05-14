#!/bin/bash

set -euo pipefail

# Usage check
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <url> [geo_accession]"
  exit 1
fi

URL="$1"

# Extract filename from URL and decode it
FILENAME_ENCODED="${URL##*/}"
FILENAME_DECODED=$(printf '%b' "${FILENAME_ENCODED//%/\\x}")
BASENAME="${FILENAME_DECODED%.gz}"
BASENAME="${BASENAME%.tar}"

# Determine GEO accession
if [[ $# -eq 2 ]]; then
  GEO_ACC="$2"
else
  GEO_ACC=$(echo "$FILENAME_DECODED" | cut -d'_' -f1 | cut -d'.' -f1)
fi

LOGFILE="logs/${GEO_ACC}.log"

# Redirect stdout and stderr to logfile
exec > >(tee -i "$LOGFILE") 2>&1

echo "=== GEO Data Download and Extraction ==="
echo "Started at: $(date)"
echo "URL: $URL"
echo "GEO Accession: $GEO_ACC"
echo "Log file: $LOGFILE"
echo "======================================="

# Define paths
RAW_DIR="../data/raw/$GEO_ACC"
UNZIP_DIR="../data/unzip/$GEO_ACC"
RAW_PATH="$RAW_DIR/$FILENAME_DECODED"
UNZIP_PATH="$UNZIP_DIR/$BASENAME"

mkdir -p "$RAW_DIR"
mkdir -p "$UNZIP_DIR"

echo "Downloading to: $RAW_PATH"
curl -L "$URL" -o "$RAW_PATH"

# Extract based on file type
if [[ "$FILENAME_DECODED" == *.tar ]]; then
  echo "Extracting tar archive to: $UNZIP_DIR"
  tar -xf "$RAW_PATH" -C "$UNZIP_DIR"
else
  echo "Unzipping to: $UNZIP_PATH"
  gunzip -c "$RAW_PATH" > "$UNZIP_PATH"
fi

echo "Done"
echo "Raw file saved to:     $RAW_PATH"
echo "Extracted file(s) saved to: $UNZIP_DIR"
echo "Finished at: $(date)"
