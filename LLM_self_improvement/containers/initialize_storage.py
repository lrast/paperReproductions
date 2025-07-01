# initilization script
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset, DatasetDict


# model initialization
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')

model.save_pretrained('llama_pretrain')
tokenizer.save_pretrained('llama_pretrain')

# dataset initialization
problems = load_dataset("openai/gsm8k", "main")


def extract_answer(row):
    row['answer'] = int(row['answer'].split('####')[-1].strip().replace(',', ''))
    return row


train_set = problems['train'].map(extract_answer)
test_set = problems['test'].map(extract_answer)

train_val_split = train_set.train_test_split(test_size=0.1)

preprocessed = DatasetDict({
                             'train': train_val_split['train'],
                             'val': train_val_split['test'],
                             'test': test_set
                            })

preprocessed.save_to_disk('gsm8k_processed')
