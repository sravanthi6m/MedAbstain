# med-llm-uncertainty-benchmark

Usage: 
```
python generate_logits.py \
  --model=<path_to_model> \
  --dataset_file=<path_to_dataset_json_file> \
  --out_dir=<path_to_output_logit_dir> \
  --prompt_methods=<base|task|shared> \
  --few_shot=<>
```

```
python calculate_uncertainty.py \
  --model=<model_name> \
  --raw_data_dir=<path_to_dataset_dir> \
  --logits_data_dir=<path_to_saved_logits> \
  --data_names=<dataset_name> \
  --prompt_methods=<base|task|shared> \
  --icl_methods=<default icl0> \
  --cal_ratio=<default 0.5> \
  --alpha=<default 0.1> \
  --out_json=<opt: path_to_output_json_file_for_results>
```

OpenAI predictions (JSONL outputs):
Run LLM inference against a dataset JSON file to produce per‑question outputs and logprobs.
Inputs are JSON arrays under `datasets/`. Outputs are JSONL under `outputs/` when using
`scripts_closed_models/python/launch_experiment.py`, or wherever you pass `--output_path`
when calling `run_experiment.py` directly.
```
cat > env/.env <<EOF
OPENAI_API_KEY=...
OPENAI_API_KEY_2=...
EOF

python run_experiment.py \
  --model gpt-4o \
  --backend openai \
  --dataset_path datasets/amboss_alldiff_train_randabst.json \
  --prompt_method shared \
  --output_path outputs/amboss_gpt4o.jsonl
```

Alternatively, run the prebuilt script (uses relative paths):
```
bash scripts_closed_models/bash/run_experiment_openai_randabst.sh
```
The `scripts_closed_models/python/launch_experiment.py` flow loads keys from `env/.env`
and writes outputs under `outputs/<dataset>/<zeroshot|fewshot>/<model>/`.

Generate perturbed datasets:
Create perturbed versions of existing datasets (e.g., to introduce abstention options).
Inputs are JSON arrays under `datasets/` and outputs are written to the path you pass
via `--perturbed_dataset` (typically under `datasets/perturbed_*.json`).
```
python quantify_uncertainty/perturbed_dataset_scripts/create_perturbed_dataset.py \
  --model gpt-4.1-mini \
  --dataset datasets/amboss_alldiff_train_noabst.json \
  --perturbed_dataset datasets/perturbed_amboss_alldiff_train_noabst.json \
  --model_key OPENAI_API_KEY
```

Batch perturbed dataset generation:
```
export OPENAI_API_KEY=...
bash scripts_closed_models/bash/run_perturbed_dataset.sh
```

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
