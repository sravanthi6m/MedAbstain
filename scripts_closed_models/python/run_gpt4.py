import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from run_experiment import run_experiment
from quantify_uncertainty.metrics_runner import (
    evaluate_outputs,
    evaluate_outputs_with_conformal,
)


load_dotenv(dotenv_path="env/.env")
api_key = os.getenv("OPENAI_API_KEY")
model_name = "Phi-4-mini" #TODO
backend = "open" #TODO

dataset_path = str(ROOT_DIR / "datasets/amboss_alldiff_train_randabst.json")  # TODO
prompt_method = "shared"
few_shot = 0
cot = False
version = "v1"

out_dir = "outputs"
run_name = f"{model_name}_amboss_randabst_{prompt_method}_fs{few_shot}_cot{cot}_{version}".lower()
output_path = f"{out_dir}/{model_name}/{run_name}.jsonl"
failures_path = f"{out_dir}/{model_name}/{run_name}_failures.jsonl"
metrics_path = f"{out_dir}/{model_name}/metrics.jsonl"

# 1. Run inference
run_experiment(
    model_name=model_name,
    backend=backend,
    dataset_path=dataset_path,
    prompt_method=prompt_method,
    few_shot=few_shot,
    cot=cot,
    output_path=output_path,
    failures_path=failures_path,
    api_key=api_key,
)

# 2. Run evaluation and save metrics
# evaluate_outputs(output_jsonl_path=output_path, output_metrics_path=metrics_path)

# after run_experiment(...)
evaluate_outputs_with_conformal(
    jsonl_path=output_path,
    out_json=metrics_path,
    prompt_method="base",  # or "shared" / "task"
    icl_method="icl0",  # keep "icl0" for now
    cal_ratio=0.3,
    alpha=0.1,
)
