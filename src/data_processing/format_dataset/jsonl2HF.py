from datasets import DatasetDict, Dataset
import json

def jsonl2DS(path):
    with open(path, 'r') as f:
        data = [json.loads(i) for i in f.readlines()]

    langs = [i['lang'] for i in data]
    inputs = [i['input'] for i in data]
    outputs = [i['output'] for i in data]
    doc_ids = [i['doc_id'] for i in data]
    orders = [i['order'] for i in data]
    file = [i['file'] for i in data]
    ds = Dataset.from_dict({'lang': langs, 'input': inputs, 'output': outputs, 'doc_id': doc_ids, 'order': orders, 'file': file})

    return ds

path='data/merged/'

# ds_dict = DatasetDict({'valid': jsonl2DS(path+'dev.jsonl')})
# ds_dict = DatasetDict({'valid': jsonl2DS('a.jsonl')})
# ds_dict.save_to_disk('data/HFDS_infer')

ds_dict = DatasetDict({'train': jsonl2DS(path+'train_few.jsonl'),
                    #    'valid': jsonl2DS(path+'dev.jsonl'),
                    #    'test': jsonl2DS(path+'test.jsonl')
                    })
ds_dict.save_to_disk('data/HFDS_5-shot')