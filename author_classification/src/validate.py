import os
import json
from pathlib import Path
from typing import Any, Literal, Dict

#типизация результата валидации
ResultKey = Literal['status', 'data']
ResultType = Dict[ResultKey, Any]


def validate(filename):
    with open(filename, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    if not isinstance(data, dict) or \
        "lang" not in data or \
        not isinstance(data['lang'], str):
        return False
    return True

def main(path: Any, passport: Any):
    with open(passport, mode='r', encoding='utf-8') as f:
        config = json.load(f)
    for dir, dirpath, filenames in os.walk(path):
        for file in filenames:
            if file not in config['files']:
                return {
                    'state': False
                }
    return {
        'state': True
    }

if __name__ == "__main__":
    path = '.data/'
    passport = './config/dataset_schemas.json'
    
    state_dict = main(path, passport)
    
    if not state_dict['state']:
        raise Exception("State is False")
            
