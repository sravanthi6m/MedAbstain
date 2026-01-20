#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="gpt-4.1-mini"
DATASET="${ROOT_DIR}/datasets/medqa_1_test_noabst.json"
PERTURBED_DATASET="${ROOT_DIR}/datasets/perturbed_medqa_1_test_noabst.json"
MODEL_KEY="OPENAI_API_KEY"
# DATASETS=("${ROOT_DIR}/datasets/medqa_1_test_noabst.json"
#  "${ROOT_DIR}/datasets/medqa_1_test_randabst.json"
#  "${ROOT_DIR}/datasets/amboss_alldiff_train_noabst.json"
#  "${ROOT_DIR}/datasets/amboss_alldiff_train_randabst.json"
# )

# PERTURBED_DATASETS=("${ROOT_DIR}/datasets/perturbed_medqa_1_test_noabst.json"
#  "${ROOT_DIR}/datasets/perturbed_medqa_1_test_randabst.json"
#  "${ROOT_DIR}/datasets/perturbed_amboss_alldiff_train_noabst.json"
#  "${ROOT_DIR}/datasets/perturbed_amboss_alldiff_train_randabst.json"
#  )

# # for i in "${!DATASETS[@]}"; do
# DATASET=${DATASETS[0]}
# PERTURBED=${PERTURBED_DATASETS[0]}

# echo "Running on dataset: $DATASET"
# echo "Outputting perturbed dataset to: $PERTURBED"
python ${ROOT_DIR}/quantify_uncertainty/perturbed_dataset_scripts/create_perturbed_dataset.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --perturbed_dataset "$PERTURBED_DATASET" \
    --model_key "$MODEL_KEY"
# done