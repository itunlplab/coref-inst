import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from unsloth import FastLanguageModel
from datasets import DatasetDict
from tqdm import tqdm
import torch
import wandb

from src.constants import LANGUAGE_DICT

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for inference configuration."""
    parser = argparse.ArgumentParser(description="Process model and instruction number.")
    parser.add_argument("--model", type=str, choices=["mistral", "llama", "gemma"],
                        required=True, help="Model name (mistral, llama, or gemma)")
    parser.add_argument("--instruction_number", type=int, choices=range(1, 7),
                        required=True, help="Instruction number (1 to 6)")
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help="Name of the dataset (e.g., HFDS_5-shot).")
    parser.add_argument('--seed', type=int, required=True, 
                        help="Random seed for reproducibility (e.g., 0).")
    parser.add_argument('--output_fp', type=str, required=True, 
                        help="Output file path")
    parser.add_argument('--model_dir', type=str, required=True, 
                        help="Model directory path")
    parser.add_argument('--ds_partition', type=str, required=True, 
                        help="Dataset partition (e.g., valid, test)")
    return parser.parse_args()

args = parse_arguments()

print(f"Selected model: {args.model}")
print(f"Instruction number: {args.instruction_number}")

model_name = args.model
num = args.instruction_number
dataset_dir = args.dataset_dir
output_fp = args.output_fp
SEED = args.seed
model_dir = args.model_dir
ds_partition = 'valid'

# mapping for phi
# mapping = None if model_name == 'llama' else {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
mapping = None

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "coref-resolution"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "true"
os.environ["WANDB_NOTEBOOK_NAME"] = "infer.py"
wandb.init(name=f'{model_name}{num}_seed{SEED}')

def format_dataset(examples: Dict[str, Any]) -> List[Tuple]:
    """Format dataset examples for inference."""
    langs = examples["lang"]
    inputs = examples["input"]
    doc_ids = examples["doc_id"]
    orders = examples["order"]
    files = examples["file"]
    dataset = []

    for lang, inp, doc_id, order, file in zip(langs, inputs, doc_ids, orders, files):
        if model_name == 'gemma':
            ids = tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': INSTRUCTION + '\n\t\n' + inp},
                ],
                mapping=mapping,
                return_tensors='pt',
                add_generation_prompt=True
            )
        else:
            ids = tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': INSTRUCTION},
                    {'role': 'user', 'content': inp},
                ],
                mapping=mapping,
                return_tensors='pt',
                add_generation_prompt=True
            )
        dataset.append((ids, inp, doc_id, lang, order, file))
    return dataset

def load_model() -> Tuple:
    """Load and configure model for inference."""
    max_seq_length = 8192
    dtype = torch.float16
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

model.generation_config.pad_token_ids = tokenizer.pad_token_id

language_dict = {
    "tr": "Turkish",
    "ca": "Catalan",
    "cs": "Czech",
    "cu": "Church Slavonic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "grc": "Ancient Greek",
    "hbo": "Ancient Hebrew",
    "hu": "Hungarian",
    "lt": "Lithuanian",
    "no": "Norwegian",
    "pl": "Polish",
    "ru": "Russian"
}

INSTRUCTION_PATH = os.path.join('..', '..', 'config', 'instructions', f'instruction{num}.txt')
with open(INSTRUCTION_PATH, 'r') as f:
    INSTRUCTION = f.read()

dataset = DatasetDict.load_from_disk(dataset_dir)
ds = dataset[ds_partition]
ds = ds.select([i for i in range(38)]) # for initial experiment
ds = format_dataset(ds)

if model_name == 'llama':
    number_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids([str(i) for i in range(1000)])).to(torch.long).to('cuda')

    mask_token_id = 50963  # tokenizer.encode('MASK')[0]
    end_token_id = 271  # tokenizer.encode('<|end_header_id|>', add_special_tokens=False)[0]
    print('MASK, END', mask_token_id, end_token_id)

    with torch.inference_mode():
        for input_ids, input_str, doc_id, lang, order, file in tqdm(ds):
            current_length = 0
            inputs_buffer = torch.zeros((1, 8192), dtype=torch.long, device='cuda')
            input_start_idx = (input_ids == end_token_id).nonzero()[2, 1]  # 0, 1
            prompt_len = input_ids.shape[-1]
            
            print("PROMPT LEN:", prompt_len)

            split_ids = (input_ids == mask_token_id).nonzero()[:, 1]  # MASK indices
            
            inputs_buffer[0, :prompt_len] = input_ids[0]
            current_length += prompt_len
            torch.set_printoptions(profile="full")

            print(tokenizer.decode(inputs_buffer[:, :current_length][0]))

            prev_const = input_start_idx
            for idx, mask_idx in enumerate(split_ids):
                const_len = mask_idx - (prev_const + 1)
                inputs_buffer[0, current_length: current_length+const_len] = input_ids[0, prev_const+1: mask_idx]
                prev_const = mask_idx
                current_length += const_len

                # tokenize the part from last MASK or begging to next MASK
                print(tokenizer.decode(inputs_buffer[:, :current_length][0]))
                response = model(inputs_buffer[:, :current_length], use_cache=True) # , past_key_values=past_key_values
                logits, past_key_values = response.logits, response.past_key_values
                next_token = number_tokens[logits[:, -1, number_tokens].argmax(axis=-1, keepdims=True)]
                inputs_buffer[0, current_length] = next_token
                current_length += 1

            
            # Add last part which is after last MASK
            length = prompt_len - prev_const - 6
            inputs_buffer[0, current_length: current_length+length] = input_ids[0, prev_const+1: prompt_len-5]
            current_length += length
            # print(tokenizer.decode(inputs_buffer[0]))
            
            outputs = inputs_buffer[0, prompt_len: current_length]
            # print(inputs.shape, inputs[prompt_len:].shape)
            f = open(output_fp, 'a')
            outputs = outputs.to(dtype=torch.int64)
            
            try:
                json.dump({'doc_id': doc_id,
                           'file': file,
                           'lang': lang,
                           'order': order,
                           'text': tokenizer.decode(outputs,
                                                    skip_special_tokens=True)
                           },
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            except Exception as e:
                json.dump({'text': str(e),
                           'tokens': str(outputs),
                           'doc_id': doc_id,
                           'lang': lang,
                           'order': order},
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            finally:
                f.close()

            del outputs, prompt_len, split_ids, current_length


elif model_name == 'phi':
    torch.set_printoptions(profile="full")
    number_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids([str(i) for i in range(10)])).to(torch.long).to('cuda')
    mask_token_id = tokenizer.encode('MASK')[0]  # 1529 16033
    end_token_id = tokenizer.convert_tokens_to_ids('<|user|>')
    # end_token_id = 32010

    with torch.inference_mode():
        for input_ids, input_str, doc_id, lang, order, file in tqdm(ds):
            current_length = 0
            inputs_buffer = torch.zeros((1, 8192), dtype=torch.long, device='cuda')
            print(input_ids, end_token_id)
            print(input_str)
            print((input_ids == 32010).shape, (input_ids == 32010))
            input_start_idx = (input_ids == 32010).nonzero()[0, 1]
            prompt_len = input_ids.shape[-1]
            
            split_ids = (input_ids == mask_token_id).nonzero()[:, 1]  # MASK indices
            
            inputs_buffer[0, :prompt_len] = input_ids[0]
            current_length += prompt_len
            
            prev_const = input_start_idx - 1
            for idx, mask_idx in enumerate(split_ids):
                const_len = mask_idx - (prev_const + 2)
                inputs_buffer[0, current_length: current_length+const_len] = input_ids[0, prev_const+2: mask_idx]
                prev_const = mask_idx
                current_length += const_len

                prev_prob = 0
                prob = 1e-8
                counter = 0
                while prob > prev_prob * 0.9:
                    # tokenize the part from last MASK or begging to next MASK
                    response = model(inputs_buffer[:, :current_length], use_cache=True)
                    logits, past_key_values = response.logits, response.past_key_values
                    next_token = number_tokens[logits[:, -1, number_tokens].argmax(axis=-1, keepdims=True)]
                    
                    prev_prob = prob
                    prob = torch.nn.functional.softmax(logits[..., -1, :], dim=-1)[..., next_token].item()

                    if prob > prev_prob * 0.9:
                        inputs_buffer[0, current_length] = next_token
                        current_length += 1
                        counter += 1
                    
                    if counter == 2: break

            outputs = inputs_buffer[0, prompt_len: current_length]
            f = open(output_fp, 'a')
            outputs = outputs.to(dtype=torch.int64)
            
            try:
                json.dump({'doc_id': doc_id,
                           'file': file,
                           'lang': lang,
                           'order': order,
                           'text': tokenizer.decode(outputs, skip_special_tokens=True)
                           },
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            except Exception as e:
                json.dump({'text': str(e),
                           'tokens': str(outputs),
                           'doc_id': doc_id,
                           'lang': lang,
                           'order': order},
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            finally:
                f.close()

            del outputs, prompt_len, split_ids, current_length


elif model_name == 'gemma':
    torch.set_printoptions(profile="full")
    number_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids([str(i) for i in range(10)])).to(torch.long).to('cuda')
    mask_token_id = tokenizer.convert_tokens_to_ids("MASK")  # 43551
    end_token_id = tokenizer.convert_tokens_to_ids("\t")
    # end_token_id = 32010

    with torch.inference_mode():
        for input_ids, input_str, doc_id, lang, order, file in tqdm(ds):
            current_length = 0
            inputs_buffer = torch.zeros((1, 8192), dtype=torch.long, device='cuda')
            print(input_ids, input_str, doc_id)
            input_start_idx = (input_ids == end_token_id).nonzero()[0, 1] + 1
            prompt_len = input_ids.shape[-1]
            
            split_ids = (input_ids == mask_token_id).nonzero()[:, 1]  # MASK indices
            
            inputs_buffer[0, :prompt_len] = input_ids[0]
            current_length += prompt_len
            
            prev_const = input_start_idx
            for idx, mask_idx in enumerate(split_ids):
                const_len = mask_idx - (prev_const + 1)
                inputs_buffer[0, current_length: current_length+const_len] = input_ids[0, prev_const+1: mask_idx]
                prev_const = mask_idx
                current_length += const_len

                prev_prob = 0
                prob = 1e-8
                counter = 0
                while prob > prev_prob * 0.9:
                    # tokenize the part from last MASK or begging to next MASK
                    response = model(inputs_buffer[:, :current_length], use_cache=True)
                    logits, past_key_values = response.logits, response.past_key_values
                    next_token = number_tokens[logits[:, -1, number_tokens].argmax(axis=-1, keepdims=True)]
                    
                    prev_prob = prob
                    prob = torch.nn.functional.softmax(logits[..., -1, :], dim=-1)[..., next_token].item()

                    if prob > prev_prob * 0.9:
                        inputs_buffer[0, current_length] = next_token
                        current_length += 1
                        counter += 1
                    
                    if counter == 2: break # decimal places

            outputs = inputs_buffer[0, prompt_len: current_length]
            f = open(output_fp, 'a')
            outputs = outputs.to(dtype=torch.int64)
            
            try:
                json.dump({'doc_id': doc_id,
                           'file': file,
                           'lang': lang,
                           'order': order,
                           'text': tokenizer.decode(outputs,
                                                    skip_special_tokens=True)
                           },
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            except Exception as e:
                json.dump({'text': str(e),
                           'tokens': str(outputs),
                           'doc_id': doc_id,
                           'lang': lang,
                           'order': order},
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            finally:
                f.close()

            del outputs, prompt_len, split_ids, current_length


elif model_name == 'mistral':
    torch.set_printoptions(profile="full")
    number_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids([str(i) for i in range(10)])).to(torch.long).to('cuda')
    mask_token_id = 10713  # tokenizer.convert_tokens_to_ids('#MASK')[0, 1]
    end_token_id = 781  # tokenizer.convert_tokens_to_ids('<0x0A>')
    # end_token_id = 32010

    with torch.inference_mode():
        for input_ids, input_str, doc_id, lang, order, file in tqdm(ds):
            current_length = 0
            inputs_buffer = torch.zeros((1, 8192), dtype=torch.long, device='cuda')

            input_start_idx = (input_ids == end_token_id).nonzero()[0, 1]
            prompt_len = input_ids.shape[-1]

            # check_1 = (1119 == input_ids)
            # check_2 = (17572 == input_ids)
            # split_ids = (check_1[:-1] & check_2[1:]).nonzero()[:, 1]
            
            split_ids = (input_ids == mask_token_id).nonzero()[:, 1]  # MASK indices
            
            inputs_buffer[0, :prompt_len] = input_ids[0]
            current_length += prompt_len
            
            prev_const = input_start_idx
            for idx, mask_idx in enumerate(split_ids):
                const_len = mask_idx - (prev_const + 1)
                inputs_buffer[0, current_length: current_length+const_len] = input_ids[0, prev_const+1: mask_idx]
                prev_const = mask_idx
                current_length += const_len

                prev_prob = 0
                prob = 1e-8
                counter = 0
                while prob > prev_prob * 0.9:
                    # tokenize the part from last MASK or begging to next MASK
                    response = model(inputs_buffer[:, :current_length], use_cache=True)
                    logits, past_key_values = response.logits, response.past_key_values
                    next_token = number_tokens[logits[:, -1, number_tokens].argmax(axis=-1, keepdims=True)]
                    
                    prev_prob = prob
                    prob = torch.nn.functional.softmax(logits[..., -1, :], dim=-1)[..., next_token].item()

                    if prob > prev_prob * 0.9:
                        inputs_buffer[0, current_length] = next_token
                        current_length += 1
                        counter += 1
                    
                    if counter == 2: break

            outputs = inputs_buffer[0, prompt_len: current_length]
            f = open(output_fp, 'a')
            outputs = outputs.to(dtype=torch.int64)
            
            try:
                json.dump({'doc_id': doc_id,
                           'file': file,
                           'lang': lang,
                           'order': order,
                           'text': tokenizer.decode(outputs, skip_special_tokens=True)
                           },
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            except Exception as e:
                json.dump({'text': str(e),
                           'tokens': str(outputs),
                           'doc_id': doc_id,
                           'lang': lang,
                           'order': order},
                          fp=f,
                          ensure_ascii=False)
                f.write('\n')
            finally:
                f.close()

            del outputs, prompt_len, split_ids, current_length

def main():
    """Main entry point for inference script."""
    # All the inference logic is already executed at module level
    # This function serves as entry point for console scripts
    with open(output_fp, 'a') as f:
        f.write('\n')
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
