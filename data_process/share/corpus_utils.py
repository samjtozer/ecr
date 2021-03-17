## some code borrowed from https://github.com/shanybar/event_entity_coref_ecb_plus

import os
import sys
from os.path import abspath, dirname
sys.path.append(dirname(abspath(__file__)))

import re
import pickle
import random
import json
import pathlib
import torch
from pprint import pprint
from tqdm import tqdm
from classes import *
from random import sample
from pathlib import Path
import xml.etree.ElementTree as ET
from nltk.tokenize.treebank import TreebankWordDetokenizer


class MentionFeatures:
    
    def __init__(self):
        pass
    
    @property
    def attrib(self):
        return self.__dict__


def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = TreebankWordDetokenizer().detokenize(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


def clear_tok(tok, default="-"):
    """清除现在文本中存在的一些特殊字符"""
    clear_token = {"\t":"", "�":"", "�":"", "\u2009":"", "½":".5"}
    for c_tok, rep_tok in clear_token.items():
        if c_tok in tok:
            tok = tok.replace(c_tok, rep_tok)
    if tok == "":
        return default
    return tok


def process_raw(file_path, return_format="doc", plus_jump=True, clear=True):
    """从xml得到文本
    
    options:
        return_format: 返回格式[token, sent, doc]
        plus_jump: plus文件中第一个句子为网址，一般需要去除
        clear: 是否处理文本token
    """
    
    if isinstance(file_path, pathlib.PosixPath):
        file_path = file_path.__str__()
    
    if not file_path.endswith(".xml"):
        file_path = file_path + ".xml"
    
    root = ET.parse(file_path).getroot()
    all_tokens = []
    for token in root.findall('token'):
        all_tokens.append(token)
    
    all_sents = {}
    for it in all_tokens:
        cur_sent_id = int(it.attrib["sentence"])
        if cur_sent_id not in all_sents:
            all_sents[cur_sent_id] = []
        all_sents[cur_sent_id].append(it)
    
    for k, v in all_sents.items():
        all_sents[k] = sorted(v, key=lambda x:int(x.attrib["number"]))
        tmp_id = 0
        for it in all_sents[k]:
            assert int(it.attrib["number"]) == tmp_id
            tmp_id += 1
    
    tmp_id = 0
    sorted_sent_ids = sorted(list(all_sents.keys()))
    for it in sorted_sent_ids:
        assert it == tmp_id
        tmp_id += 1
    
    text = []
    for it in sorted_sent_ids:
        if plus_jump and "plus" in file_path and it == 0:
            continue
        if clear:
            text.append([clear_tok(x.text) for x in all_sents[it]])
        else:
            text.append([x.text for x in all_sents[it]])
    
    if return_format == "token":
        return text
    elif return_format == "sent":
        return [untokenize(x) for x in text]
    elif return_format == "doc":
        return " ".join([untokenize(x) for x in text])


def get_head_lemma(nlp, x):
    """获取mention的head word和其lemma
    
    输入:
        nlp: 传入的spacy对象
        x: 传入的mention

    输出:
        返回mention的head和其对应的lemma
    """
    x_parsed = nlp(x)
    for tok in x_parsed:
        if tok.head == tok:
            if tok.lemma_ == u'-PRON-':
                return str(tok.text), str(tok.text)
            return str(tok.head), str(tok.lemma_)


def format_doc(all_docs, doc_id, untoken=True, output=False):
    """返回处理后的corpus中存在的文档数据

    输入:
        all_docs: 输入的所有文档字典
        doc_id: 文档名ID
    输出:
        文档中的句子字典: key为句子ID, values为句子内容
    """
    res = {}
    sent_ids = all_docs[doc_id].sentences.keys()
    sent_ids = sorted(sent_ids)
    for ID in sent_ids:
        if untoken:
            string = untokenize(all_docs[doc_id].sentences[ID].get_tokens_strings())
        else:
            string = " ".join(all_docs[doc_id].sentences[ID].get_tokens_strings())
        res[ID] = string
        if output:
            print(f"{ID}:  {string}" + "\n")
    return res


def topic_to_mention_list(topic, is_gold=True):
    """给定一个topic对象，返回其中所有的event mention和entity mention"""
    event_mentions = []
    entity_mentions = []
    for doc_id, doc in topic.docs.items():
        for sent_id, sent in doc.sentences.items():
            if is_gold:
                event_mentions.extend(sent.gold_event_mentions)
                entity_mentions.extend(sent.gold_entity_mentions)
            else:
                event_mentions.extend(sent.pred_event_mentions)
                entity_mentions.extend(sent.pred_entity_mentions)

    return event_mentions, entity_mentions


def find_head_index(mention):
    """返回指定mention对象的head word对应的token id(句子内的)"""
    for token in mention.tokens:
        if mention.mention_head.lower() == token.get_token().lower() or \
           mention.mention_head.lower() in token.get_token().lower():
            return int(token.token_id)
    return None


def docID2topicID(string):
    """返回doc ID对应的topicID"""
    index = string.find("_")
    topic = string[:index] + "_ecb"
    if "plus" in string:
        topic += "plus"
    return topic


def doc2topic(string):
    """返回doc ID对应的topicID"""
    index = string.find("_")
    topic = string[:index] + "_ecb"
    if "plus" in string:
        topic += "plus"
    return topic


def mention2sent(data, mention):
    """返回包含该mention的sent实例
    
    输入:
        data: 对应包含该mention的corpus对象(比如train_data, test_data, dev_data)
              或者所有文档的all_docs对象{doc_id: doc}
        mention: mention对象实例
    
    输出:
        包含该mention的sentence对象实例
    """

    topic_id = doc2topic(mention.doc_id)
    doc_id, sent_id = mention.doc_id, mention.sent_id
    if type(data) == dict:
        return data[doc_id].sentences[sent_id]
    else:
        return data.topics[topic_id].docs[doc_id].sentences[sent_id]

def mention2doc(all_docs, mention):
    """返回包含该mention的doc实例
    
    输入:
        all_docs: 包含所有文档的字典
        mention: mention对象实例
    """
    return all_docs[mention.doc_id]


def sentID2sent(all_docs, sent_id):
    """返回sent_id对应的sentence对象
    
    输入:
        all_docs: 包含所有文档的字典
        sent_id: 示例(1_10ecb_3)(1_10ecb中的第3个句子)
    """
    tmp = sent_id.split("_")
    doc_id, sent_id = "_".join(tmp[:-1]), int(tmp[-1])
    return all_docs[doc_id].sentences[sent_id]


def load_predicted_topics(test_data, predicted_topic_list):
    """将文档聚类的结果转变为topic实例"""
    new_topics = {}
    
    all_docs = []
    for topic in test_data.topics.values():
        all_docs.extend(topic.docs.values())
    all_doc_dict = {doc.doc_id:doc for doc in all_docs }

    topic_counter = 1
    for topic in predicted_topic_list:
        topic_id = str(topic_counter)
        new_topics[topic_id] = Topic(topic_id)

        for doc_name in topic:
            if doc_name in all_doc_dict:
                new_topics[topic_id].docs[doc_name] = all_doc_dict[doc_name]
        topic_counter += 1

    return new_topics