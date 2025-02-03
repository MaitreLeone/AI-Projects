import os
import json
from typing import List
import requests
import math
import extractive_summarization
from nltk.tokenize import sent_tokenize

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

model_name = "Upstage-Llama-2-70B-instruct-v2-GPTQ"

configs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
host = {
    'Upstage-Llama-2-70B-instruct-v2-GPTQ': '',
    'Platypus2-70B-Instruct-GPTQ': ''
}

other_name_lang = {
    "ru":"russian",
    "en":"english",
    "de":"",
    "fr":"",
    "es":"",
    "ar":"",
    "tr":"",
    "uk":""
}

top_k = 50
top_p = 0.95
mean_len_sent_tokens = 100
max_text_len_char = 4000

def init():
    # Чтение шаблона затравки из конфигурационного файла
    global string_for_split, lang_for_template, prompt_templates, instruction_templates, sentence_for_template
    if os.path.exists(configs_path):
        with open(os.path.join(configs_path), 'r') as f:
            configs = json.load(f)[model_name]['prompt_template']
        string_for_split = configs['string_for_split']
        lang_for_template = configs['lang_for_template']
        prompt_templates = configs['prompts_input']
        instruction_templates = configs['instruction_template']
        sentence_for_template = configs['sentence_for_template']
    else:
        print("Don't exist config for llm summarization")
        raise

def llm_summarization(text:str, language:str, texts_count:int, max_len:int, instruction:str) -> List[str]:
    # Запрос к модели для получения реферата
    prompt_template = prompt_templates[language]
    prompt = prompt_template.format(instruction=instruction, text=text)

    data = {
        'prompt': prompt,
        'params': {
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': True,
            'max_new_tokens': max_len*mean_len_sent_tokens,
            'num_return_sequences': texts_count
        }
    }

    response = requests.post(host[model_name], json=data, proxies={"http": '', "https": ''})
    if response.status_code == 200:
        result = [item["text"].split(string_for_split)[1].strip() for item in json.loads(response.text)]
        return result
    return []

def instruction_create(language:str, max_len:int) -> str:
    # Формирование конкретных инструкций с учетом требуемых языка и длины
    lang = lang_for_template[language]
    instruction_template = instruction_templates[language]
    if language == "ru":
        word = morph.parse(sentence_for_template[language])[0]
        max_length = f"{max_len} {word.make_agree_with_number(max_len).word}"
    elif language == "en":
        if max_len == 1:
            max_length = f"one sentence"
        else:
            max_length = f"{max_len} sentences"
    # elif ... or only english prompt?
    
    instruction = instruction_template.format(language=lang, max_length=max_length)
    
    return instruction

def main(text:str, language:str, text_count:int=1, max_len:int=3) -> List[str]:
    # Разделение текста на блоки. Формирование инструкции. Реферирование каждого блока с обращением к LLM. Удаление незаврешенных, лишних предложений 
    # Добавить формирование реферата для длинного текста (если len(text) > [3600-4000], обращение к экстрактивному реферированию)
    input_text = text
    instruction = instruction_create(language, max_len)
    if len(text) >= max_text_len_char:
        compression = len(text)/max_text_len_char
        # В экстрактивном методе или коэффициент сжатия, или предложения
        # num_sentences = len(sent_tokenizer(text))
        # target_num_sentences = int(num_sentences/compression)
        input_text = extractive_summarization.main(text, language, compression)[0]
    summaries = llm_summarization(input_text, language, text_count, max_len, instruction)
    output_summaries = []
    for summary in summaries:
        # Разделяем реферат на предложения
        sent_summary = sent_tokenize(summary, other_name_lang[language]) #wr_segmentation.do([{'text':summary, 'lang':language}], min_sent_len=1)[0]['sentences']
        # Пока длина реферата превышает max_characters, и в реферате больше одного предложения, удалется последнее предложение
        while len(sent_summary) > max_len:
            sent_summary = sent_summary[:-1]
            summary = " ".join([sent for sent in sent_summary])
        output_summaries.append(summary)
    return output_summaries

if __name__ == "__main__":
    init()
    text = ""
    language="ru"
    texts_count = 2
    max_length = 3 
    print(main(text=text, language=language, text_count=texts_count, max_len=max_length))