#!/bin/bash
# Upload training data to SAIS via S3
# Usage: bash research/scripts/upload_to_sais.sh

set -e

S3_BUCKET="s3://chess-coach-training-data"
DATA_DIR="research/data"

echo "=== Uploading training data to S3 ==="

# Lichess studies (primary Stage 2 data)
if [ -f "$DATA_DIR/lichess_studies.jsonl" ]; then
    LINES=$(wc -l < "$DATA_DIR/lichess_studies.jsonl")
    echo "Lichess studies: $LINES pairs"
    aws s3 cp "$DATA_DIR/lichess_studies.jsonl" "$S3_BUCKET/lichess_studies.jsonl" --profile chess-deck
else
    echo "WARNING: lichess_studies.jsonl not found"
fi

# Chess concepts (alignment data)
if [ -f "$DATA_DIR/chess_concepts_10k.jsonl" ]; then
    LINES=$(wc -l < "$DATA_DIR/chess_concepts_10k.jsonl")
    echo "Chess concepts: $LINES pairs"
    aws s3 cp "$DATA_DIR/chess_concepts_10k.jsonl" "$S3_BUCKET/chess_concepts_10k.jsonl" --profile chess-deck
fi

# Training script
if [ -f "research/encoder/scripts/train_lichess_stage2.py" ]; then
    echo "Training script"
    aws s3 cp "research/encoder/scripts/train_lichess_stage2.py" "$S3_BUCKET/train_lichess_stage2.py" --profile chess-deck
fi

echo ""
echo "=== Upload complete ==="
echo "On SAIS, download with:"
echo "  aws s3 cp $S3_BUCKET/lichess_studies.jsonl /home/ec2-user/SageMaker/chess-research/data/"
echo "  aws s3 cp $S3_BUCKET/chess_concepts_10k.jsonl /home/ec2-user/SageMaker/chess-research/data/"
