# -*- coding: utf-8 -*-
import os
os.environ['http_proxy'] = "http://word:word1word@172.16.200.240:8080"
os.environ['https_proxy'] = "http://word:word1word@172.16.200.240:8080"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

# Commented out IPython magic to ensure Python compatibility.
# %pip install transformers datasets

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
import pyarrow as pa
from statistics import mean
from dvclive.huggingface import DVCLiveCallback
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.get_device_name(0)

name_dir = './ready_datasets/train_test/'
name_file_train_sent = 'train_authors_texts.csv'
name_file_val_sent = 'val_authors_texts.csv'
name_file_test_sent = 'test_authors_texts.csv'

random_state = 42
test_size = 0.2 #размер тестовой части
val_size = 0.2 #размер валидационной части
max_len = 512 #максимальная длина входной последовательности
decrease_tokens = 100 #количество токеов на которое уменьшается максимальная длина до достижения порогового значения
part_clipped_texts = 0.01 #пороговое значение количества обрезанных текстов
model_name = '/opt/models/multilingual-e5-large'
path_to_model = '/opt/models/multilingual-e5-large'

#Training arguments
output_dir='/opt/models/multilingual_e5_large/checkpoints(25.01)/'
overwrite_output_dir=True
save_strategy='epoch'
num_train_epochs=30
per_device_train_batch_size=32
per_device_eval_batch_size=32
weight_decay=0.01
learning_rate=1e-5
logging_dir='./logs'
evaluation_strategy='epoch'

"""# Загрузка корпуса авторских предложений"""

path_to_authors_stat = './additional_info/'

with open(path_to_authors_stat + 'authors_dict.json', mode='r', encoding='utf-8') as f:
    authors_dict = json.load(f)[0]

num_author = len(list(authors_dict.keys()))

authors_id_dict = {v: k for k, v in authors_dict.items()}

df_train = pd.read_csv(name_dir+name_file_train_sent, sep='\t')
df_val = pd.read_csv(name_dir+name_file_val_sent, sep='\t')
df_test = pd.read_csv(name_dir+name_file_test_sent, sep='\t')

"""# Размер данных"""

tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_token_len(df, tokenizer):
    token_lens = []
    # for txt in df.sentence:
    for txt in df.text:
        tokens = tokenizer.encode(txt)
        token_lens.append(len(tokens))
    return token_lens

def token_len(token_lens, max_len):
    number_outliers = sum(length > max_len for length in token_lens)
    print(f'min length = {min(token_lens)} tokens')
    print(f'mean length = {mean(token_lens)} tokens')
    print(f'max length = {max(token_lens)} tokens')
    print(f'{number_outliers} texts longer than {max_len} tokens ({number_outliers/len(token_lens)})')
    fig = sns.histplot(token_lens)
    plt.xlabel('Token count')
    plt.show()

def define_max_len(df_train, df_val, df_test, tokenizer, max_len):
    token_lens_train = get_token_len(df_train, tokenizer)
    token_lens_val = get_token_len(df_val, tokenizer)
    token_lens_test = get_token_len(df_test, tokenizer)
    shorten = True
    while shorten:
        number_outliers_train = sum(length > max_len for length in token_lens_train)
        number_outliers_val = sum(length > max_len for length in token_lens_val)
        number_outliers_test = sum(length > max_len for length in token_lens_test)
        part_outliers_train = number_outliers_train/len(df_train)
        part_outliers_val = number_outliers_val/len(df_val)
        part_outliers_test = number_outliers_test/len(df_test)
        if part_outliers_train < part_clipped_texts and part_outliers_val < part_clipped_texts and part_outliers_test < part_clipped_texts:
            max_len -= decrease_tokens
        else:
            max_len += decrease_tokens
            shorten = False
    print('data type: train')
    token_len(token_lens_train, max_len)
    print('\ndata type: validation')
    token_len(token_lens_val, max_len)
    print('\ndata type: test')
    token_len(token_lens_test, max_len)
    print(f'\ndefine max_len = {max_len}')
    return max_len

max_len_value = define_max_len(df_train, df_val, df_test, tokenizer, max_len)

"""# Подготовка данных"""

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
test_dataset = Dataset.from_pandas(df_test)

print(f'train_dataset:\n\n{train_dataset}')

def tokenize(batch):
    if max_len_value <= max_len:
        return tokenizer(
            ['query: ' + text for text in batch['text']],
            max_length=max_len_value,
            padding='max_length',
            truncation='longest_first'
        )
    else:
        return tokenizer(
            ['query: ' + text for text in batch['text']],
            padding='max_length',
            truncation='longest_first'
        )

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

"""# Загрузка модели и обучение"""

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_author)
model.config.label2id = authors_dict
model.config.id2label = authors_id_dict

def compute_metrics(pred):
    '''Функция возвращает метрики качества'''
    labels = pred.label_ids
    preds = [p.argmax() for p in pred.predictions]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    metrics = {
        'accuracy': acc,
        'f1_mean': np.mean(f1),
        'f1': f1.tolist(),
        'precision_mean': np.mean(precision),
        'precision': precision.tolist(),
        'recall_mean': np.mean(recall),
        'recall': recall.tolist()
    }
    for key, value in metrics.copy().items():
        if isinstance(value, list):
            for i, metric in enumerate(value):
                metrics[f'{key}_{i}'] = metric
            metrics.pop(key)
    return metrics

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    save_strategy=save_strategy,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    logging_dir=logging_dir,
    evaluation_strategy=evaluation_strategy,
    save_total_limit = 1,
    load_best_model_at_end = True
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)

trainer.add_callback(DVCLiveCallback(log_model="all"))
trainer.train_dataset = train_dataset
trainer.eval_dataset = val_dataset

start_time = time()

trainer.train()

print(f'Time: {time() - start_time:.1f} sec')