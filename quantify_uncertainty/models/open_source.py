from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel
import torch.nn.functional as F


class OpenSourceHFModel(BaseModel):
    def __init__(self, model_name: str, max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.model_max_length = min(
            self.tokenizer.model_max_length, max_length
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _prepare_inputs(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        return {k: v.to("cuda") for k, v in inputs.items()}

    def generate(self, prompt: str, choices: List[str]):
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

        output_text = self.tokenizer.decode(
            output_tokens.sequences[0], skip_special_tokens=True
        )
        logits = output_tokens.scores[-1].squeeze(0)
        logprobs = F.log_softmax(logits, dim=-1)

        choice_token_map = {
            c: self.tokenizer.encode(c, add_special_tokens=False) for c in choices
        }
        for c, token_ids in choice_token_map.items():
            if len(token_ids) > 1:
                print(
                    f"⚠️ Warning: Choice '{c}' tokenized into multiple tokens: {token_ids}"
                )

        selected_logprobs = {
            c: float(logprobs[token_ids[0]].item())
            for c, token_ids in choice_token_map.items()
        }
        return {
            "output": output_text.strip(),
            "raw_logprobs": selected_logprobs,
            "option_keys_for_logits": choices,
        }

    def get_logits(self, prompt: str, choices: List[str]) -> List[float]:
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits[:, -1, :].squeeze(0)

        option_tokens = [
            self.tokenizer.encode(f"Answer: {k}", add_special_tokens=False)[-1]
            for k in choices
        ]
        return logits[option_tokens].float().cpu().numpy().tolist()
