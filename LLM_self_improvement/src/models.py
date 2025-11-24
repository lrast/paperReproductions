# Models for self-training
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def initialize_model_and_tokenizer(target_directory,
                                   base_model='meta-llama/Llama-3.2-1B-Instruct',
                                   sft_template='src/chat_template.j2',
                                   **kwargs):
    """Setup model and tokenizer in target directory"""
    model = AutoModelForCausalLM.from_pretrained(base_model)
    target_directory = Path(target_directory)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
    model.resize_token_embeddings(len(tokenizer))

    if sft_template:
        # modify the chat template to allow the SFT trainer to be used 
        with open(sft_template) as f:
            template = f.read()
            tokenizer.chat_template = template

    model.save_pretrained(target_directory)
    tokenizer.save_pretrained(target_directory)


def load_model_and_tokenizer(directory):
    """Load model and tokenizer from target directory"""
    directory = Path(directory)

    model = AutoModelForCausalLM.from_pretrained(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    return model, tokenizer
