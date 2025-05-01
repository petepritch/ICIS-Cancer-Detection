#!/bin/bash
#SBATCH --job-name=isic-training
#SBATCH --partition=spgpu
#SBATCH --account=eecs545w25_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-user=petep@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log

# Load modules
module load python/3.9.12
module load cuda/11.6.2
module load cudnn

# Initialize conda
source /home/petep/ICIS-Cancer-Detection/isic_env/bin/activate

export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
export CONFIG_DIR="$PROJECT_ROOT/config"
export DATA_DIR="/scratch/your_uniqname/tumor_data"
export OUTPUT_DIR="$PROJECT_ROOT/models"
export LOG_DIR="$PROJECT_ROOT/logs"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "GPU information:"
nvidia-smi

PROJECT_ROOT=$(pwd)
CONFIG_DIR="$PROJECT_ROOT/config"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/models"
LOG_DIR="$PROJECT_ROOT/logs"
EMBEDDING_DIR="$PROJECT_ROOT/embeddings"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$EMBEDDING_DIR"

if [ ! -f "$DATA_DIR/metadata_preprocessed.csv" ]; then
    echo "Error: Data not found. Please make sure your data is in the correct location."
    exit 1
fi

cd /home/petep/ICIS-Cancer-Detection

echo "Training vision model..."
python scripts/train_model.py \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --log-file "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

if [ -f "/home/petep/ICIS-Cancer-Detection/isic_env/bin/deactivate" ]; then 
	source /home/petep/ICIS-Cancer-Detection/isic_env/bin/deactivate
fi

VISION_MODEL_PATH="$OUTPUT_DIR/$(ls -t $OUTPUT_DIR | grep -v "lgbm" | head -1)"
MODEL_NAME=$(grep "architecture" "$CONFIG_DIR/model_config.yaml" | cut -d: -f2 | tr -d ' "')
EMBEDDING_NAME="${MODEL_NAME}_$(date +%Y%m%d)"

echo "Extracting embeddings from $VISION_MODEL_PATH..."
python scripts/extract_embeddings.py \
    --config-dir "$CONFIG_DIR" \
    --model-path "$VISION_MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --output-dir "$EMBEDDING_DIR" \
    --csv-file "$DATA_DIR/metadata_preprocessed.csv" \
    --img-dir "$DATA_DIR/train-image/image" \
    --embedding-name "$EMBEDDING_NAME" \
    --log-file "$LOG_DIR/extract_embeddings_$(date +%Y%m%d_%H%M%S).log"

# 3. Train LGBM model using embeddings
echo "Training LGBM model..."
python scripts/train_lgbm.py \
    --config-file "$CONFIG_DIR/lgbm_config.yaml" \
    --metadata-file "$DATA_DIR/metadata_preprocessed.csv" \
    --embedding-dir "$EMBEDDING_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-names "$EMBEDDING_NAME" \
    --log-file "$LOG_DIR/train_lgbm_$(date +%Y%m%d_%H%M%S).log"

echo "Pipeline completed successfully!"