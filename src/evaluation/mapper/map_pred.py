#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module converts mention tags in text to ConLL-U formatted mentions with
the same entity group.
"""

import re
import os
import json
import copy
import logging
import argparse

from collections import OrderedDict
import udapi.core.coref
from udapi.block.write.conllu import Conllu as ConlluWriter
import udapi.block.corefud.removemisc


def mwt_serialize(words):
    """_summary_

    Args:
        words (_type_): _description_

    Returns:
        _type_: _description_
    """
    for w in words:
        if isinstance(w, list):
            idx = words.index(w)
            for item in w[::-1]:
                words.insert(idx, item)
            words.remove(w)

    return words


def match_mentions(mention_words, words):
    """_summary_

    Args:
        mention_words (_type_): _description_
        words (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Assume start and end are the same
    # Consider MWTs
    words = mwt_serialize(words)
    # try:
    #     assert bool(words)
    # except AssertionError as e:
    #     print(mention_words, words)
    #     return
   
    alternatives = list()
    found_end = None
    
    for gold_words, gold_head in mention_words:
        if words[0]._root._bundle._bundle_id != gold_words[0]._root._bundle._bundle_id:
            continue

        gold_start_address = str(gold_words[0].address)\
            if not gold_words[0].multiword_token\
            else str(gold_words[0].multiword_token.words[0].address)
        
        if gold_words[-1].multiword_token:
            gold_end_address = str(
                gold_words[-1].multiword_token.words[-1].address
            )
        # elif gold_words[-1].is_empty() and len(gold_words) > 1:
        #     i = 0
        #     while len(gold_words) > -i and not gold_words[i].is_empty():
        #         i -= 1
        #         gold_end_address = str(gold_words[i].address)
        # else:
        #     gold_end_address = str(gold_words[-1].address)
        else:
            gold_end_address = str(gold_words[-1].address)

        # if words[-1].is_empty() and len(words) > 1:
        #     i = 0
        #     while len(words) > -i and not words[-i].is_empty():
        #         i -= 1
        #         found_end = str(words[i].address)
        # else:
        #     found_end = str(words[-1].address)
        found_end = str(words[-1].address)
        
        # Direct match
        if gold_start_address == str(words[0].address) and \
            found_end and \
           gold_end_address == found_end:
            return gold_words, gold_head

        # End match
        elif found_end and gold_end_address == found_end:
            alternatives.append((gold_words, gold_head))

    if alternatives:
        if len(alternatives) == 1:
            return alternatives[0]
        else:
            return min(
                alternatives, 
                key=lambda i: abs(i[0][0].ord - words[0].ord)
            )  # first word dist
    
    raise ValueError(f"Not matched {words}")


def map_to_udapi(doc, clusters_independent, prev_ent_counter, mention_words):
    """_summary_

    Args:
        doc (_type_): _description_
        clusters_independent (_type_): _description_
        prev_ent_counter (_type_): _description_
        mention_words (_type_): _description_

    Returns:
        _type_: _description_
    """
    udapi_words = list(doc.nodes_and_empty)
    mention_words = [
        (
            [
                [j for j in list(doc.nodes_and_empty) if j.address() == w][0]
                for w in i[0]
            ],
            [j for j in list(doc.nodes_and_empty) if j.address() == i[1]][0]
        )
        for i in mention_words
    ]
    
    if len(clusters_independent) != len(mention_words):
        logging.debug('Not same number of mentions!')

    # Remove MWT single nodes
    for node in udapi_words:
        if node.multiword_token:
            idx = udapi_words.index(node)
            udapi_words.insert(idx, list(node.multiword_token.words))
            for mwt in node.multiword_token.words:
                udapi_words.remove(mwt)

    # Convert from key, value to group of mentions
    clusters = dict()
    for k, v in clusters_independent.items():
        v += prev_ent_counter
        if v in clusters:
            clusters[v].append(k)
        else:
            clusters[v] = [k]

    # Create mentions on UDAPI document
    for cluster_id in clusters:
        entity = doc.create_coref_entity(eid='e' + str(cluster_id))
        for (start, end) in clusters[cluster_id]:
            if start < 0:
                start = 0
            words = udapi_words[start:end]

            matched_words, matched_head = match_mentions(mention_words, words)
            mention_words.remove((matched_words, matched_head))
            try:
                entity.create_mention(head=matched_head, words=matched_words)
            except ValueError as e:
                print(e)
                print(matched_words, words)
            
    udapi.core.coref.store_coref_to_misc(doc)
    return doc


def is_empty_mention(node):
    for mention in node.coref_mentions:
        if len(mention.words) == 1:
            return True
    return False


class Mapper:
    """_summary_
    """
    def __init__(self, doc, llm_output_windows):
        """_summary_

        Args:
            doc (_type_): _description_
            llm_output_windows (_type_): _description_
        """
        self.doc = doc

        self.llm_windows = [
            (window1, window2) for window1, window2 in llm_output_windows
        ]
        self.entity_counter = 0
        self.clusters = {}
        self.window_clusters = []
        
        self.udapi_word_forms = []
        last_mwt = None
        for node in self.doc.nodes_and_empty:
            if node.multiword_token:
                if last_mwt == node.multiword_token:
                    continue
                self.udapi_word_forms.append(node.multiword_token.form)
                last_mwt = node.multiword_token
            
            # elif node.is_empty() and not is_empty_mention(node):
            #     self.udapi_word_forms.append('_')

            elif node.is_empty():
                if node.coref_mentions:
                    if is_empty_mention(node):
                        self.udapi_word_forms.append('__')
                    else:
                        self.udapi_word_forms.append('')
                else:
                    self.udapi_word_forms.append('')

            else:
                self.udapi_word_forms.append(node.form)

        self.look = copy.deepcopy(self.udapi_word_forms)

    def map_outputs(self, prev_ent_counter, gold_conllu_fp):
        """_summary_

        Args:
            prev_ent_counter (_type_): _description_
            gold_conllu_fp (_type_): _description_
        """
        start_w_1 = 0

        for row in self.llm_windows:
            self.udapi_word_forms, clusters_1, start_w_1 = \
                self.create_clusters(self.udapi_word_forms, row[0], start_w_1)
            udapi_word_forms_2 = copy.copy(self.udapi_word_forms)
            start_w_2 = copy.copy(start_w_1)
            udapi_word_forms_2, clusters_2, start_w_2 = self.create_clusters(
                udapi_word_forms_2, row[1], start_w_2
            )
            self.window_clusters.append((clusters_1, clusters_2))

        self.merge_clusters()

        # Remove entity misc and re-read the document
        mention_words = [
            ([w.address() for w in i.words], i.head.address())
            for i in self.doc.coref_mentions
        ]
        
        doc_cpy = udapi.Document(gold_conllu_fp, encoding="utf-8")
        udapi.block.corefud.removemisc.RemoveMisc(
            attrnames=(
                "Entity,SplitAnte,Bridge"
            )
        ).apply_on_document(doc_cpy)
        udapi.block.write.conllu.Conllu(
            files=['dev_help.conllu']
        ).apply_on_document(doc_cpy)
        doc = udapi.Document('dev_help.conllu')

        del self.doc
        self.doc = map_to_udapi(
            doc, self.clusters, prev_ent_counter, mention_words
        )

    def merge_clusters(self):
        """_summary_

        Raises:
            ValueError: _description_
        """
        self.clusters = self.window_clusters[0][0]
        # first window before [MID]
        if self.clusters:
            counter = max(self.clusters.values())
        else:
            counter = 0

        for before, after in self.window_clusters:
            before = dict(sorted(before.items(), key=lambda i: i[0][0]))
            after = dict(sorted(after.items(), key=lambda i: i[0][0]))
            mapping = dict()

            for k, v in before.items():
                if k not in self.clusters:
                    raise ValueError(f"WINDOWS ARE NOT SAME {k}, {v}")
                mapping[v] = self.clusters[k]

            for k, v in after.items():
                if v in mapping:
                    self.clusters[k] = mapping[v]
                    if self.clusters[k] > counter:
                        counter = self.clusters[k]
                else:
                    # New entity
                    counter += 1
                    mapping[v] = counter
                    self.clusters[k] = counter

    def create_clusters(self, udapi_word_forms, window, starting_word_order):
        """_summary_

        Args:
            udapi_word_forms (_type_): _description_
            window (_type_): _description_
            starting_word_order (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
            e: _description_

        Returns:
            _type_: _description_
        """
        w_id = starting_word_order
        mention_starts = []
        new_clusters = {}

        while window:
            window = window.strip()
            if udapi_word_forms:
                form = udapi_word_forms[0]
            else:
                form = 'None'  # There are still some mentions at the end

            try:
                if w_id < len(self.look) and self.look[w_id] != form:
                    raise ValueError("Not aligned!")
                
                if window.startswith('<m>'):
                    window = window[len('<m>'):]
                    mention_starts.append(w_id)
                
                elif window.startswith('</m>#'):
                    window = window[len('</m>#'):]
                    match = re.match(r'^\d+', window)
                    cluster_id = match.group() if match else None
                    window = window[len(cluster_id):]
                    start_id = mention_starts.pop()

                    # ill-contioned mentions
                    if (start_id, w_id) in new_clusters:
                        new_clusters[(start_id-1, w_id)] = int(cluster_id)

                    else:
                        new_clusters[(start_id, w_id)] = int(cluster_id)

                elif form == '':  # empty but there is no mention
                    w_id += 1
                    window = window[len('_'):]
                    udapi_word_forms.pop(0)

                elif window.startswith('</z>@') and form == '__':
                    window = window[len('</z>@'):]
                    match = re.match(r'^\d+', window)
                    cluster_id = match.group() if match else None
                    window = window[len(cluster_id):]
                    new_clusters[(w_id, w_id+1)] = int(cluster_id)
                    w_id += 1
                    udapi_word_forms.pop(0)  # pop __
                    
                elif window.lower().startswith(form.lower()):
                    window = window[len(form):]
                    w_id += 1
                    udapi_word_forms.pop(0)
                
                elif window == '':
                    break
                
                else:
                    print(form)
                    print(window)
                    raise ValueError("Unexpected sequence.")

            except Exception as e:
                print(form)
                print(window)
                raise e

        return udapi_word_forms, new_clusters, w_id


def write_data(udapi_docs, f):
    """_summary_

    Args:
        udapi_docs (_type_): _description_
        f (_type_): _description_
    """
    writer = ConlluWriter(filehandle=f)
    for doc in udapi_docs:
        writer.before_process_document(doc)
        writer.process_document(doc)
    writer.after_process_document(None)


def write_docs_to_files(directory, filename):
    """_summary_

    Args:
        directory (_type_): _description_
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Write docs to different Conllu files to read faster
    if not os.path.exists(directory):
        os.makedirs(directory)

    doc_order = list()

    with open(filename, 'r') as f:
        docs = re.split(r'(?=# newdoc)', f.read())[1:]

        for text in docs:
            doc_id = text.split('\n')[0].split('=')[1].strip()
            doc_order.append(doc_id)
            doc_id = doc_id.replace('/', '%2')
            doc_path = f'{directory}/{doc_id}+{os.path.basename(filename)}'
            with open(doc_path, 'w') as doc_file:
                doc_file.write(''.join(text))

    return doc_order


def main(filename, output_file, raw_windows, gold_conllu):
    """_summary_

    Args:
        filename (_type_): _description_
        output_file (_type_): _description_
        raw_windows (_type_): _description_
    """
    s = 'mapper/'

    lang = raw_windows[0]['lang']
    file_name = raw_windows[0]['file']
    ds_name = file_name.split('-')[0]
    output_file = os.path.join(
        os.path.dirname(output_file), 
        ds_name + '-' + os.path.basename(output_file)
    )

    # Seperate document lines
    docs = {}
    for line in raw_windows:
        if line['doc_id'] in docs:
            docs[line['doc_id']].append(line)
        else:
            docs[line['doc_id']] = [line]

    # Map and write data
    prev_max_ent = 0
    docs_dir = 'mapper/docs'
    doc_order = write_docs_to_files(docs_dir, filename)

    docs = OrderedDict((key, docs[key]) for key in doc_order)

    if os.path.exists(output_file):
        os.remove(output_file)

    for doc_id, lines in docs.items():
        # sort windows by order
        docs[doc_id] = sorted(lines, key=lambda i: i['order'])

        # Read original Conllu file
        doc_id = doc_id.replace('/', '%2')
        doc_file_name = f'{doc_id}+{os.path.basename(filename)}'
        udapi_doc = udapi.Document(os.path.join(docs_dir, doc_file_name))

        # Map text to Conllu
        for window in lines:
            if "[MID]" not in window['text']:
                window['text'] = window['text'] + "[MID]"

        lines_text = [
            tuple(map(str.strip, window['text'].split('[MID]')))
            for window in lines
        ]  # text
        mapper = Mapper(udapi_doc, lines_text)
        mapper.map_outputs(prev_max_ent, os.path.join(docs_dir, doc_file_name))
        if mapper.doc.coref_entities:
            prev_max_ent = max(
                list(map(lambda i: int(i.eid[1:]), mapper.doc.coref_entities))
            )
            prev_max_ent += len(mapper.doc.coref_entities)

        with open(output_file, "a", encoding="utf-8") as f:
            write_data([mapper.doc], f)

        # SCORE DOCUMENTS #
        doc_score = f"mapper/doc_score+{doc_id}.conllu"
        doc_gold = f"mapper/gold_doc+{doc_id}.conllu"
        with open(doc_score, "w", encoding="utf-8") as f:
            write_data([mapper.doc], f)

        start_line = "# newdoc id = " + mapper.doc.meta['docname'] + '\n'
        with open(filename, 'r', encoding="utf-8") as f:
            doc_lines = list()
            flag = False
            while True:
                line = f.readline()
                if start_line == line:
                    flag = True
                elif line.startswith("# newdoc id = "):
                    flag = False

                if flag:
                    doc_lines.append(line)
                if not line:
                    break

        with open(doc_gold, 'w', encoding='utf-8') as f:
            f.write(''.join(doc_lines))

        print(mapper.doc.meta['docname']+'\n')
        os.remove(doc_gold)
        os.remove(doc_score)
        # SCORE DOCUMENTS #

    print(lang, file_name)
    os.system(
        (
            f'python3 {s}tools/validate.py --level 2 --coref '
            f'--lang {lang} {output_file}'
        )
    )
    os.system(
        f'python3 {s}corefud-scorer/corefud-scorer.py {gold_conllu} {output_file}'
    )
    print(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gold_conllu', default='mapper/dev.conllu',
                        required=False, help='Gold conllu data to compare')
    parser.add_argument('-p', '--donor_conllu', default='mapper/dev.conllu',
                        required=False,
                        help=('Predicted mentions conllu data for mention '
                              'location'))
    parser.add_argument('-o', '--output_path', default="mapper/out.conllu",
                        required=False, help="output path")
    parser.add_argument('-l', '--llm_output_file',
                        default='mapper/output.jsonl', required=False,
                        help="LLM jsonl çıktısı")
    parser.add_argument('-f', '--filename',
                        required=False,
                        help="Name of the dataset without extension")
    parser.add_argument('-d', '--doc', required=False,
                        help="doc_id for getting one doc from jsonl")
    parser.add_argument('-s', '--dataset_name', required=False,
                        help="for example: tr_itcc")
    args = parser.parse_args()
    print(args.gold_conllu, args.output_path, args.llm_output_file)

    llm_output = list()
    if args.donor_conllu:
        donor_conllu = args.donor_conllu
    else:
        donor_conllu = args.gold_conllu

    print("LLM out:", args.llm_output_file)
    if args.dataset_name:
        print("With dataset name")
        with open(args.llm_output_file, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = json.loads(line)
                if args.dataset_name in line['file']:
                    llm_output.append(line)
        print(len(llm_output))

    elif args.filename:
        with open(args.llm_output_file, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = json.loads(line)
                if line['file'] == args.filename + '.jsonl':
                    llm_output.append(line)

    elif args.doc:
        with open(args.llm_output_file, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line or not line.strip():
                    break
                line = json.loads(line)
                if line['doc_id'] == args.doc:
                    llm_output.append(line)

    else:
        with open(args.llm_output_file, 'r', encoding='utf-8') as f:
            try:
                llm_output = [json.loads(line) for line in f.readlines()]
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except IOError as e:
                print(f"I/O error: {e}")
            except ValueError as e:
                print(f"Value error: {e}")

    assert llm_output
    main(donor_conllu, args.output_path, llm_output, args.gold_conllu)
