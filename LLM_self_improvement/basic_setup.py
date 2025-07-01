from transformers import pipeline
from datasets import load_dataset


ds = load_dataset("openai/gsm8k", "main")
pipe = pipeline("text-generation", model="NousResearch/Llama-3.2-1B",
                do_sample=True, temperature=1.)
