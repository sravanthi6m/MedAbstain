from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-V3"
ROOT_DIR = Path(__file__).resolve().parents[2]
save_path = str(ROOT_DIR / "models")
# Load model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Save locally
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
