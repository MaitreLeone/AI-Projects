import os
import re
from razdel import tokenize, sentenize
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split


dvc_root = os.environ['DVC_ROOT']

def read_json(path):
    file_obj = open(path, encoding="utf-8")
    data = json.load(file_obj) 
    file_obj.close()
    return data

def read_files(path):
    all_texts = []
    authors = []
    for address, dirs, files in os.walk(path):
        author_texts = []
        #authors.append(dirs)
        for name in files:
            s = os.path.join(address, name)
            file_text = read_json(s)
            author_texts.extend(file_text)
        all_texts.append(author_texts)
    return all_texts

def preprocess(all_texts):
    texts = []
    authors = []
    uniq_authors = []
    count_author = 0
    count_all_texts = []
    count_long_texts = []
    for author in all_texts:
        count_all_text = 0
        count_long_text = 0
        for file in author:
            aut = file['meta']['authors'][0]
            if aut not in uniq_authors:
                uniq_authors.append(aut)
            text = file['text']
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"youtube.com\S+", "", text)
            text = re.sub(r"t.me\S+", "", text)
            text = re.sub(r"instagram.com\S+", "" ,text)
            text = re.sub(r"\S*@\S*\s?", "", text)
            text = re.sub(r"[@#]\S*\s?", "", text)
            text = re.sub('\*+', '', text)
            text = re.sub('==+', '', text)
            text = re.sub(' +', ' ', text)
            text = re.sub('\s{2,}','\n', text)
            authors.append(aut)
            texts.append(text)
        '''
            for block in text.split("\n"):
                if 'sentences' in file:
                    len_text_block = len(file['sentences'])
                else:
                    len_text_block = len(list(sentenize(block)))
                number_words = len(block.split(" "))
                if len_text_block > 2 and len_text_block < 7 and number_words > 14: #and detect(block) == 'ar':
                    authors.append(aut)
                    texts.append(block)
                    count_long_text += 1
                count_all_text += 1
        count_author += 1
        count_all_texts.append(count_all_text)
        count_long_texts.append(count_long_text)
        '''
    return texts, authors


def get_labeled_dataset(texts, authors, path='./ready_datasets/', save=True):
    sent_all_autors = {}
    sent_all_autors['text'] = texts
    sent_all_autors['author'] = authors
    df_all_sent = pd.DataFrame(sent_all_autors)
    if save:
        if not os.path.exists(path):
            os.mkdir(path)
        df_all_sent.to_csv(path + "all_selected_authors_text.csv", index=False, sep='\t')

def process_files(path='./data/'):
    all_texts = read_files(path)
    texts, authors = preprocess(all_texts)
    if not os.path.exists('ready_datasets'):
        os.mkdir('ready_datasets')
    get_labeled_dataset(texts, authors)
	
process_files()
