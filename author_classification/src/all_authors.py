import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

path_to_info = './additional_info/'
path_to_config = './config/'

def get_all_authors(path='./ready_datasets/', save=True):
    df_all_authors_sent = pd.DataFrame()
    for dir, dirpath, files in os.walk(path):
        for file in files:
            try:
                df_all_sent = pd.read_csv(path + file, sep='\t')
            except:
                df_all_sent = pd.read_csv(path + file)
            df_all_authors_sent = pd.concat([df_all_authors_sent, df_all_sent])
    #df_all_authors_sent = df_all_authors_sent.drop(columns=['label'])
    if save:
        df_all_authors_sent.to_csv(path + 'all_authors_sent.csv')
    df_all_authors_sent = df_all_authors_sent.reset_index()
    df_all_authors_sent = df_all_authors_sent.drop(columns=['index'])
    possible_labels = df_all_authors_sent.author.unique()
    possible_labels.sort()
    label_dict = {}
    with open(path_to_config + 'metrics_schemas.json', mode='r', encoding='utf-8') as f:
        metrics = json.load(f)
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = int(index)
        
        metrics[0][f'precision_{index}'] = f'Точность_{possible_label}'
        metrics[0][f'recall_{index}'] = f'Полнота_{possible_label}'
        metrics[0][f'f1_{index}'] = f'F1_{possible_label}'
    with open(path_to_config + 'metrics_schemas.json', mode='w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    label_list = [label_dict]

    if not os.path.exists(path_to_info):
        os.mkdir(path_to_info)
    with open(path_to_info + 'authors_dict.json', mode='w', encoding='utf-8') as f:
        json.dump(label_list, f, indent=2, ensure_ascii=False)

    df_all_authors_sent['label'] = df_all_authors_sent.author.replace(label_dict)
    n_texts = int(np.round(np.median(df_all_authors_sent['author'].value_counts())))
    df_select_sent = df_all_authors_sent[df_all_authors_sent['label']==0].sample(frac=1)[:n_texts]
    for i in range(1, len(possible_labels)):
        df_select_sent = pd.concat([df_select_sent, df_all_authors_sent[df_all_authors_sent['label']==i].sample(frac=1)[:n_texts]])
    return df_select_sent

def get_train_test_datasets(df_select_sent, path='./ready_datasets/train_test/', save=True):
    X, X_val, y, y_val = train_test_split(df_select_sent.index.values,
                                      df_select_sent.label.values,
                                      test_size=0.1,
                                      random_state=42,
                                      stratify=df_select_sent.label.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.1,
                                        random_state=42,
                                        stratify=y)
    df_select_sent['data_type'] = ['not_set']*df_select_sent.shape[0]

    df_select_sent.loc[X_train, 'data_type'] = 'train'
    df_select_sent.loc[X_val, 'data_type'] = 'val'
    df_select_sent.loc[X_test, 'data_type'] = 'test'

    stat_df = pd.DataFrame(df_select_sent.groupby(['author', 'label', 'data_type']).count())
    stat_df.to_csv(path_to_info + 'stat.csv', sep='\t')

    if save:
        if not os.path.exists(path):
            os.mkdir(path)
        (df_select_sent.loc[X_train][['text', 'author', 'label']]).to_csv(path + "train_authors_texts.csv", index=False, sep='\t')
        (df_select_sent.loc[X_val][['text', 'author', 'label']]).to_csv(path + "val_authors_texts.csv", index=False, sep='\t')
        (df_select_sent.loc[X_test][['text', 'author', 'label']]).to_csv(path + "test_authors_texts.csv", index=False, sep='\t')

df_select_sent = get_all_authors()
get_train_test_datasets(df_select_sent)
