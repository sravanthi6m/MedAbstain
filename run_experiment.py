import argparse
import json
import os
from tqdm import tqdm

from quantify_uncertainty.prompts.prompt_helpers import PROMPT_DISPATCH
from quantify_uncertainty.models.open_source import OpenSourceHFModel
from quantify_uncertainty.models.openai_model import OpenAIModel
from quantify_uncertainty.data_helpers.loaders import load_all_data
from quantify_uncertainty.dynamic_sampler import (
    DynamicSampler,
    SentenceTransformerModel,
    OpenAIEmbeddingModel,
)

def run_experiment(
    model_name: str,
    backend: str,
    dataset_path: str,
    prompt_method: str = "base",
    few_shot: int = 0,
    cot: bool = False,
    output_path: str = "experiment_outputs.jsonl",
    failures_path: str = "experiment_failures_record.jsonl",
    api_key: str = None,
    dataset_type: str = None,
    few_shot_pool_1: str = None,
    embedding_model: str = None,
    few_shot_pool_2: str = None,
):
    dynamic_fewshot_map = None
    if few_shot > 0:
        print("Generating dynamic samples")

        if not few_shot_pool_1 or not os.path.exists(few_shot_pool_1):
            raise FileNotFoundError(
                "Dynamic few-shot requires a valid path to json file at "
                f"--few_shot_pool_1. Path not found: {few_shot_pool_1}"
            )

        if "text-embedding" in embedding_model:
            embedding_model_instance = OpenAIEmbeddingModel(model_name=embedding_model)
        else:
            embedding_model_instance = SentenceTransformerModel(
                model_name=embedding_model
            )

        sampler = DynamicSampler(
            data_file_path=dataset_path,
            few_shot_pool_path_1=few_shot_pool_1,
            few_shot_pool_path_2=few_shot_pool_2,
            embedding_model=embedding_model_instance,
        )
        dynamic_fewshot_map = sampler.get_dynamic_few_shot_examples(
            k=few_shot, pool_1_percentage=1.0
        )

    raw_data = load_all_data(
        os.path.dirname(dataset_path),
        os.path.splitext(os.path.basename(dataset_path))[0],
    )
    format_fn = PROMPT_DISPATCH[prompt_method]

    if backend == "open":
        model = OpenSourceHFModel(model_name)
    elif backend == "openai":
        model = OpenAIModel(model_name, api_key=api_key)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as fout, open(failures_path, "a") as ferr:
        for ex in tqdm(raw_data, desc=f"{model_name} | {prompt_method}"):
            current_fewshot_examples = (
                dynamic_fewshot_map.get(ex["id"], []) if dynamic_fewshot_map else None
            )
            formatted = format_fn(
                ex,
                argparse.Namespace(k_few_shot=few_shot, cot=cot),
                current_fewshot_examples,
            )
            prompt = formatted["prompt"]
            choices = list(ex["choices"].keys())

            meta = {
                "model": model_name,
                "few_shot": few_shot,
                "cot": cot,
                "dataset_type": dataset_type,
            }
            result = {
                "id": ex["id"],
                "source": ex["source"],
                "prompt": prompt,
                "choices": choices,
                "answer": ex["answer"],
                "meta": meta,
            }
            try:
                model_response = model.generate(prompt, choices)
                result["output"] = model_response["output"]
                result["logprobs"] = model_response["raw_logprobs"]

                fout.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"⚠️ Error on ID {ex['id']}: {e}")
                fail_record = {**result, "error": str(e)}
                ferr.write(json.dumps(fail_record) + "\n")

    print(f"Saved outputs to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True, choices=["open", "openai"])
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--prompt_method", default="base", choices=["base", "shared", "task"])
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--output_path", default="experiment_outputs.jsonl")
    parser.add_argument("--failures_path", default="experiment_failures_record.jsonl")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--dataset_type", default=None)
    parser.add_argument("--few_shot_pool_1", default=None)
    parser.add_argument("--few_shot_pool_2", default=None)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        model_name=args.model,
        backend=args.backend,
        dataset_path=args.dataset_path,
        prompt_method=args.prompt_method,
        few_shot=args.few_shot,
        cot=args.cot,
        output_path=args.output_path,
        failures_path=args.failures_path,
        api_key=args.api_key,
        dataset_type=args.dataset_type,
        few_shot_pool_1=args.few_shot_pool_1,
        embedding_model=args.embedding_model,
        few_shot_pool_2=args.few_shot_pool_2,
    )
