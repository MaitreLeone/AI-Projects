import json
import os
import io
import gzip
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

def iterate_one_text(elem, marking_dict_reverse, samples_list):
    # обработка одного текста
    
    tmp_topic = []
    if 'st_topics' in elem['meta']:
        for topic in elem['meta']['st_topics']:
            if topic in marking_dict_reverse[elem['lang']]['st_topics']:
                tmp_topic.append(marking_dict_reverse[elem['lang']]['st_topics'][topic])

    if 'ex_topics' in elem['meta']:
        for topic in elem['meta']['ex_topics']:
            if topic in marking_dict_reverse[elem['lang']]['ex_topics']:
                tmp_topic.append(marking_dict_reverse[elem['lang']]['ex_topics'][topic])

    if tmp_topic != []:
        samples_list.append({'text' : elem['text'], 'lang': elem['lang'], 'topics': list(set(tmp_topic))})
            
    return samples_list


def split_data(samples_list, rubrics):
    # деление на обучающую, валидационную и тестовую выборки
    
    for i in range(len(samples_list)):
        samples_list[i]['split_str'] = ''
        for rub in rubrics:
            if rub in samples_list[i]['topics']:
                samples_list[i]['split_str'] += '1'
            else:
                samples_list[i]['split_str'] += '0'
        
        samples_list[i]['split_str'] += samples_list[i]['lang']
        
    freq_dict = {} # удалим варианты комбинации "топики - язык" с менее чем 10 примерами
    for elem in samples_list:
        if elem['split_str'] not in freq_dict:
            freq_dict[elem['split_str']] = 0
        freq_dict[elem['split_str']] += 1
        
    lowfreq_list = []
    for key in freq_dict:
        if freq_dict[key] < 10:
            lowfreq_list.append(key)

    filtered_samples_list = []
    for elem in samples_list:
        if elem['split_str'] not in lowfreq_list:
            sample = {'text': elem['lang'] + ' : ' + elem['text'], 'split_str': elem['split_str']}
            for rub in rubrics:
                if rub in elem['topics']:
                    sample[rub] = 1
                else:
                    sample[rub] = 0
            filtered_samples_list.append(sample)

    df = pd.DataFrame.from_dict(filtered_samples_list)
    X, X_valid, y, y_valid = train_test_split(df, df['split_str'], test_size=0.1, random_state=42, stratify=df['split_str'])
    X_train, X_test, y_train, y_test = train_test_split(X, X['split_str'], test_size=0.1, random_state=42, stratify=X['split_str'])
    
    return X_train, X_valid, X_test



def preparate(path='data', marking_dict_path='src/utils/topicMarkingDict.json'):
    # функция подготовки файлов к обучению
    # сохраняет файлы в формате .arrow в папке prepared_dataset
    
    with open(marking_dict_path, mode='r', encoding='utf-8') as f:
        marking_dict = json.load(f)
        
    marking_dict_reverse = {} # меняем ключ (topic) и значение (st/ex topic) местами 
    for lang in marking_dict:
        marking_dict_reverse[lang] = {}
        for st_ex_mark in marking_dict[lang]:
            marking_dict_reverse[lang][st_ex_mark] = {}
            for key in marking_dict[lang][st_ex_mark]:
                for elem in marking_dict[lang][st_ex_mark][key]:
                    marking_dict_reverse[lang][st_ex_mark][elem] = key
    
    samples_list = []
    
    for dir, dirpath, filenames in os.walk(path):
        for file in filenames:
            if file.endswith('.json'): # если в формате .json
                with open(os.path.join(dir, file), mode='r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file.endswith('.json.gz'):
                with gzip.open(os.path.join(dir, file), 'r') as f: # если в формате .json.gz
                    data = json.loads(f.read().decode('utf-8'))

            if isinstance(data, list): # если данные в формате списка словарей в одном файле
                for elem in data:
                    samples_list = iterate_one_text(elem, marking_dict_reverse, samples_list)

            elif isinstance(data, dict): # если данные в формате словаря в одном файле
                samples_list = iterate_one_text(data, marking_dict_reverse, samples_list)
                
    train_df, valid_df, test_df = split_data(samples_list, list(marking_dict[lang][st_ex_mark].keys()))

    train_dataset = datasets.Dataset.from_pandas(train_df)
    eval_dataset = datasets.Dataset.from_pandas(valid_df)
    test_dataset = datasets.Dataset.from_pandas(test_df) 
    dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset, "validation":eval_dataset})

    if not os.path.exists('prepared_dataset'):
        os.makedirs('prepared_dataset')
    
    dataset.save_to_disk('prepared_dataset')
    
    return 

preparate()
