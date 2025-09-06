import argparse
import json
import os
import re
import shutil
from typing import List, Tuple, Dict, Any

import udapi
from transformers import AutoTokenizer, logging

logging.set_verbosity_error()

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def frag_long_sentences(sent: str, max_length: int, tokenizer: AutoTokenizer) -> Tuple[int, List[int], List[str]]:
    # Break up long sentences
    frag_sents = [sent]
    frag_lengths = [max_length+1] # initial state
    
    # if any fragment is longer than max_length fragment
    while sum([l > max_length for l in frag_lengths]) > 0:
        for idx, length in enumerate(frag_lengths):
            if length > max_length:
                mid_sent = len(frag_sents[idx]) // 2
                frag_sents[idx:idx+1] = [frag_sents[idx][:mid_sent], frag_sents[idx][mid_sent:]]
                frag_lengths = [len(tokens) for tokens in tokenizer.batch_encode_plus(frag_sents).input_ids]

    return len(frag_lengths), frag_lengths, frag_sents

def dynamic_document_to_alpaca(tokenizer, file, file_name, sentences, window_size=1500, doc_id=None):
    """
    This function dynamically sums up the sentences according to token lengths up until half of window_size.
    Half of the window is taken from the previous content.
    Returns: {order: <beggining index of the first sentence>,
            'input': '</m>#MASK',
            'output': '</m>#0'}
    """
    max_length = window_size // 2

    tokens = tokenizer.batch_encode_plus(sentences)
    lengths = list(map(lambda i: len(i), tokens.input_ids))

    index = 0
    start = 0
    discrete_window_indices = []
    long = False
    while len(lengths) != 0:
        # Create windows shorter than max_length
        current_sum = 0
        start = index
        
        if lengths[0] > max_length:
            # Pass long sentences
            # print(lengths[0])
            # lengths.pop(0)

            # Add long sentencens as fragments
            n, lengths[0:1], sentences[start:start+1] = frag_long_sentences(sentences[start], max_length, tokenizer)
            index += n
            long = True
            
            # All of the fragments are long enough to be added one by one
            for i in range(n):
                discrete_window_indices.append((start, index+i))
                lengths.pop(0)
                start = index
            continue

        while lengths and current_sum + lengths[0] <= max_length:
            current_sum += lengths.pop(0)
            index += 1
        discrete_window_indices.append((start, index))

    if len(discrete_window_indices) == 1: discrete_window_indices.append((0,0))

    lang = file_name.split('/')[-1].split('_')[0]
    file_dataset = []
    for i in range(len(discrete_window_indices)-1):
        window = ' '.join(sentences[discrete_window_indices[i][0]: discrete_window_indices[i][1]]) + \
        '[MID]' + ' '.join(sentences[discrete_window_indices[i+1][0]: discrete_window_indices[i+1][1]])
        pattern = r'(</[mz]>[@#])(\d+)'
        comp = re.compile(pattern)

        input_ = comp.sub(r'\1MASK', window)
        mentions = [groups[1] for groups in comp.findall(window)]
        
        ordered_set = []
        for j in mentions:
            if j not in ordered_set:
                ordered_set.append(j)

        d = {i: str(idx) for idx, i in enumerate(ordered_set)}
        output = comp.sub(lambda m: f"{m.group(1)}{d[m.group(2)]}", window)

        json.dump({'lang': lang,
                   'order': discrete_window_indices[i][0],
                   'long_in_doc': str(long),
                   'file': file_name.split('/')[-1],
                   'doc_id': doc_id,
                   'len': len(tokenizer(f'{input_}[MID]{output}')[0]),
                   'input': input_,
                   'output': output}, file, ensure_ascii=False)
        file.write('\n')

    return file_dataset

def document_to_alpaca(file_name, full_text, stride, window_size):
    sentences = full_text.split('\n')

    file_dataset = []
    n = len(sentences)

    for i in range(n // stride):
        start = i * stride
        end = start + window_size if n > start + window_size else n

        window = ''.join(sentences[start: end])

        pattern = r'(</[mz]>[@#])(\d+)'
        comp = re.compile(pattern)
        
        input_ = comp.sub(r'\1MASK', window)

        mentions = comp.findall(window)

        ordered_set = []
        for j in mentions:
            if j not in ordered_set:
                ordered_set.append(j)

        d = {i: str(idx) for idx, i in enumerate(ordered_set)}

        output = comp.sub(lambda m: f"{m.group(1)}{d[m.group(2)]}", window)
        
        file_dataset.append({'file': file_name, 'order': i, 'window': window ,'input': input_, 'output': output})
    
    return file_dataset

def is_empty_mention(node):
    for mention in node.coref_mentions:
        if len(mention.words) == 1:
            return True
        else: pass
    return False


def add_mention_tag(doc):
    for node in doc.nodes_and_empty:
        if node.is_empty():
            if is_empty_mention(node):
                node.form = ''
            else:
                node.form = '_'

    for mention in doc.coref_mentions[::-1]:
        start = mention.words[0]
        end = mention.words[-1]

        if start.is_empty() and start == end:
            start.form += f"</z>@{mention.entity.eid[1:]} "
        else:
            if start.multiword_token:
                start.multiword_token.form = "<m> " + start.multiword_token.form
            else:
                start.form = "<m> " + start.form
            
            if end.multiword_token:
                end.multiword_token.form += f" </m>#{mention.entity.eid[1:]}"
            else:
                end.form += f" </m>#{mention.entity.eid[1:]}"
            

    return doc

def compute_text(doc):
    result = []
    for tree in doc.trees:
        last_mwt_id = 0
        string = ''
        for node in tree.descendants_and_empty:
            mwt = node.multiword_token
            if mwt:
                # pass tokens up until last mwt
                # add the mwt
                if node._ord > last_mwt_id:
                    last_mwt_id = mwt.words[-1]._ord
                    string += mwt.form
                    if mwt.misc['SpaceAfter'] != 'No':
                        string += ' '
            
            elif node.is_empty():
                string += node.form
                parent = node.deps[0]['parent']
                if parent.misc['SpaceAfter'] != 'No':
                    string += ' '
                                
            else:
                string += node.form
                if node.misc['SpaceAfter'] != 'No':
                    string += ' '
        result.append(string.rstrip())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', required=True, help='Kaynak göstermek için dosya adı')
    parser.add_argument('-p', '--path', required=True, help="örnek: tr/itcc/tr.conllu")
    parser.add_argument('-o', '--output_path', required=True, help="örnek: /home/etc/1.jsonl")
    parser.add_argument('-w', '--window_size', default=1500, type=int, help="Penceredeki cümle sayısı")
    parser.add_argument('-s', '--stride', default=20, type=int, help="Atlanacak cümle sayısı")
    args = parser.parse_args()

    assert args.window_size > args.stride

    if 'train' in args.path:
        opt = 'train'
    elif 'dev' in args.path:
        opt = 'dev'
    elif 'test' in args.path:
        opt = 'test'

    dir = f'{os.path.dirname(args.path)}/{opt}_documents'
    if not os.path.exists(dir): os.makedirs(dir)

    with open(args.path, 'r') as f:
        text = f.read()
        docs = re.split(r'(?=# newdoc)', open(args.path, 'r').read())[1:]
        
        for idx, text in enumerate(docs):
            with open(f'{dir}/{idx}+{os.path.basename(args.path)}', 'w') as doc_file:
                doc_file.write(''.join(text))

    tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=None)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for fp in os.listdir(dir):
            doc_dir = os.path.join(dir, fp)
            
            doc = udapi.Document(filename=doc_dir)
            
            mention_doc = add_mention_tag(doc)
            sents = compute_text(mention_doc)
            dynamic_document_to_alpaca(tokenizer, f, args.output_path, sents, args.window_size, doc.meta['docname'])

    shutil.rmtree(dir)
    print(args.output_path)