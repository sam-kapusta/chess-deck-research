#!/bin/bash
# Download Qwen3-4B-Instruct and upload to SAIS via S3.
#
# Prerequisites:
#   - AWS credentials for account 140023406996 (ada)
#   - pip install huggingface_hub
#
# Usage:
#   ./setup_qwen_on_sais.sh
#
# This will:
#   1. Download Qwen3-4B-Instruct (~8GB) locally
#   2. Upload to S3 bucket
#   3. Then download on SAIS via MCP (much faster, internal network)

set -e

MODEL="Qwen/Qwen3.5-4B"
LOCAL_DIR="/tmp/qwen3-4b"
S3_BUCKET="temp123123312"
S3_PREFIX="chess-research/models/qwen3-4b"

echo "=== Qwen3-4B Setup for SAIS ==="

# Step 1: Download locally
echo "[1/3] Downloading $MODEL..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$LOCAL_DIR', ignore_patterns=['*.gguf', '*.bin'])
print('Downloaded!')
"

# Step 2: Upload to S3
echo "[2/3] Uploading to s3://$S3_BUCKET/$S3_PREFIX..."
aws s3 sync "$LOCAL_DIR" "s3://$S3_BUCKET/$S3_PREFIX" --exclude "*.gguf"

echo "[3/3] Done! On SAIS, run:"
echo "  aws s3 sync s3://$S3_BUCKET/$S3_PREFIX /home/ec2-user/SageMaker/chess-research/models/qwen3-4b"
echo ""
echo "Or generate a presigned URL and curl from SAIS."
