import re
import csv
import sys
import json
import torch
import pickle
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from random import sample
from pprint import pprint

sys.path.append("../share")
from classes import *
from corpus_utils import topic_to_mention_list, load_predicted_topics, \
    mention2sent, mention2doc, doc2topic, find_head_index

arg_parser = argparse.ArgumentParser(description="generate pair-wise samples from pickled data")

arg_parser.add_argument("--seed", type=int, default=2020, help="set the random seed")
arg_parser.add_argument("--data_dir", type=str, default="../data/pkl_data", help="the directory of pickled data")
arg_parser.add_argument("--predicted_topics", type=str, default="../data/topics/predicted_topics", help="the predicted topic file")
arg_parser.add_argument("--output_dir", type=str, default="../data/pairwise", help="the directory of output files")
arg_parser.add_argument("--tsv_dir", type=str, default="../data/tsv_data", help="the directory of output files")

args = arg_parser.parse_args()

def writeCSV(pairs, columns, file_path, delimiter="\t"):
    with open(file_path, "w", encoding="UTF-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)


def get_bert_seqclass_pairs(topics, ratio=0.5):
    """given a topic, generate all mention-pairs in this topic"""
    
    global all_docs
    
    input_pairs = []
    for topic in tqdm(topics):
        mentions, _ = topic_to_mention_list(topic)
        for m_i in mentions:
            mention1_tokens = m_i.tokens_numbers
            
            mention1_tokens = sorted(mention1_tokens)
            i_s, i_e = mention1_tokens[0], mention1_tokens[-1]
            sent_1 = mention2sent(all_docs, m_i).get_tokens_strings()
            ori_sent_1 = sent_1.copy()
            sent_1 = sent_1[:i_s] + ["<coref>"] + sent_1[i_s:i_e+1] + ["<coref>"] + sent_1[i_e+1:]
            
            mention1_id = m_i.mention_id
            for m_j in mentions:
                
                is_coref = 1 if m_i.gold_tag == m_j.gold_tag else 0

                if not is_coref and random.random() > ratio:
                    continue
                
                mention2_tokens = m_j.tokens_numbers
                
                mention2_tokens = sorted(mention2_tokens)
                j_s, j_e = mention2_tokens[0], mention2_tokens[-1]
                sent_2 = mention2sent(all_docs, m_j).get_tokens_strings()
                ori_sent_2 = sent_2.copy()
                sent_2 = sent_2[:j_s] + ["<coref>"] + sent_2[j_s:j_e+1] + ["<coref>"] + sent_2[j_e+1:]
                
                mention2_id = m_j.mention_id
                
                
                tmp = {
                    "sent1": " ".join(sent_1),
                    "sent2": " ".join(sent_2),
                    "ori_sent1": ori_sent_1,
                    "ori_sent2": ori_sent_2,
                    "is_coref": str(is_coref),
                    "mention1_id": mention1_id,
                    "mention2_id": mention2_id,
                    "mention1_tokens": mention1_tokens,
                    "mention2_tokens": mention2_tokens
                }
                input_pairs.append(tmp)
    return input_pairs


def writeJson(data, file_path):
    """each line is a json object"""
    with open(file_path, "w", encoding="utf-8") as file:
        for it in tqdm(data):
            file.write(json.dumps(it, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":

    random.seed(args.seed)

    ## recover the pickle dumped data
    data_dir = Path(args.data_dir)

    with open(data_dir / "train_data", 'rb') as f:
        train_data = pickle.load(f)

    with open(data_dir / "dev_data", 'rb') as f:
        dev_data = pickle.load(f)

    with open(data_dir / "test_data", 'rb') as f:
        test_data = pickle.load(f)

    ## get all docs(obtain doc using doc_id)
    all_docs = {}
    for topic in train_data.topics.values():
        for doc in topic.docs.values():
            all_docs[doc.doc_id] = doc
    for topic in dev_data.topics.values():
        for doc in topic.docs.values():
            all_docs[doc.doc_id] = doc
    for topic in test_data.topics.values():
        for doc in topic.docs.values():
            all_docs[doc.doc_id] = doc
    
    ## use the document clustering result from https://github.com/shanybar/event_entity_coref_ecb_plus
    with open(args.predicted_topics, "rb") as file:
        predicted_topics = pickle.load(file)

    tmp_topics = load_predicted_topics(test_data, predicted_topics)


    ## get event-pair samples
    seqclass_train_inputs = get_bert_seqclass_pairs(train_data.topics.values(), ratio=0.5)
    seqclass_train_all_inputs = get_bert_seqclass_pairs(train_data.topics.values(), ratio=1)
    seqclass_dev_inputs = get_bert_seqclass_pairs(dev_data.topics.values(), ratio=1)
    seqclass_test_gold_inputs = get_bert_seqclass_pairs(test_data.topics.values(), ratio=1)
    seqclass_test_pred_inputs = get_bert_seqclass_pairs(tmp_topics.values(), ratio=1)

    ## store all samples
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    writeJson(seqclass_train_inputs, output_dir / "train.json")
    writeJson(seqclass_train_all_inputs, output_dir / "train_all.json")
    writeJson(seqclass_dev_inputs, output_dir / "dev.json")
    writeJson(seqclass_test_pred_inputs, output_dir / "test_pred.json")
    writeJson(seqclass_test_gold_inputs, output_dir / "test_gold.json")

    ## stored as tsv format files
    tsv_dir = Path(args.tsv_dir)
    if not tsv_dir.exists():
        tsv_dir.mkdir()

    columns = ["is_coref", "mention1_id", "mention2_id", "sent1", "sent2"]

    writeCSV(seqclass_train_inputs, columns, tsv_dir / "train.tsv")
    writeCSV(seqclass_train_all_inputs, columns, tsv_dir / "train_all.tsv")
    writeCSV(seqclass_dev_inputs, columns, tsv_dir / "dev.tsv")
    writeCSV(seqclass_test_gold_inputs, columns, tsv_dir / "test_gold.tsv")
    writeCSV(seqclass_test_pred_inputs, columns, tsv_dir / "test_pred.tsv")
    writeCSV(seqclass_test_pred_inputs, columns, tsv_dir / "test.tsv")