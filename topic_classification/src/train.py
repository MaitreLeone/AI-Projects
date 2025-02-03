import os
import io
import json
import datasets
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from joblib import dump, load

from dvclive import Live

def train_models(path='prepared_dataset', learning_rate=0.001):
    # функция обучения модели MLPClassifier на предобработанном датасете
    
    vectorizer = HashingVectorizer(n_features=10000)
    
    dataset = datasets.load_from_disk(path)
    
    if os.path.exists('config/params.json'):
        with open('config/params.json', mode='r', encoding='utf-8') as f:
            params = json.load(f)
        if 'learning_rate' in params:
            learning_rate = params['learning_rate']
    
    with Live() as live:

        live.log_param("скорость обучения", learning_rate)

        X_train = vectorizer.fit_transform(dataset['train']['text'])
        X_valid = vectorizer.fit_transform(dataset['validation']['text'])

        list_categories = []
        list_scores = []

        for category in dataset['train'].column_names[2:-1]:

            y_train = list(dataset['train'][category])
            y_valid = list(dataset['validation'][category])

            clf = MLPClassifier(learning_rate_init=learning_rate, random_state=42)
            clf.fit(X_train, y_train)

            y_valid_pred = clf.predict(X_valid)
            score = f1_score(y_valid, y_valid_pred)
            live.log_metric("eval_f1", score)
            list_scores.append(score)

            if not os.path.exists('trained'):
                os.makedirs('trained')

            dump(clf, 'trained/' + category + '-mlp.joblib')
            list_categories.append(category)
            
        live.log_metric("eval_f1", sum(list_scores) / len(list_scores))
        list_categories.append('Mean')

    eval_tsv_path = 'dvclive/plots/metrics/eval_f1.tsv'
    
    if os.path.exists(eval_tsv_path):
        eval_df = pd.read_csv(eval_tsv_path, sep='\t')

        eval_df['categories'] = list_categories
        eval_df.to_csv(eval_tsv_path, sep="\t") 

    return

train_models()