import requests
import json
import time
from datetime import datetime
import base64
import logging
import os
from tqdm import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def login(api_url, username, password):
    credentials = base64.b64encode(f"{username}:{password}".encode('utf-8')).decode('utf-8')
    response = requests.post(f"{api_url}/sign_in",
                             headers={'accept': 'application/json', 'Authorization': f'Basic {credentials}'})

    if response.status_code == 200 or response.status_code == 201:
        token = response.json().get("access_token")
        return token
    else:
        logging.error(f"Ошибка авторизации для пользователя {username}. Код ответа: {response.status_code}")
        logging.error("Текст ответа: %s", response.text)
        return None


def clear_chat_info(api_url, chat_id, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(f"{api_url}/chat/{chat_id}/history", headers=headers)

    if response.status_code == 200 or response.status_code == 201 or response.status_code == 204:
        logging.info(f"История чата {chat_id} успешно очищена")
    else:
        logging.error(f"Ошибка при очистке истории чата {chat_id}. Код ответа: {response.status_code}")


def save_results_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


# Загрузка конфигурации из файла config.yaml
with open("config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Настройка кастомного хоста и модели
api_key = config["api_key"]
base_url = config["base_url"]

# Создание клиента OpenAI с кастомным хостом и API ключом
client = openai.OpenAI(api_key=api_key, base_url=base_url)


def process_question(api_url, chat_id, token, qa_pair, config, index):
    question = qa_pair["question"]
    ground_truth = qa_pair["ground_truth"]
    url = f'{api_url}/search/{chat_id}'
    error_counter = 0

    while error_counter < 5:
        try:
            r = requests.post(url, headers={'Authorization': f'Bearer {token}'}, json={
                "text": question,
                "return_debug_data": True,
                "meta_override": config["test_config"],
                "debug_true": True
            })
            time.sleep(1)  # Уменьшаем время ожидания между запросами
            response_data = r.json()

            answer = response_data.get("response")
            context = "\n".join([ctx["content"] for ctx in response_data["debug_data"]["reranked_context"]])

            result = {
                "Question Number": index + 1,
                "Question": question.strip(),
                "Answer": answer,
                "Context": context,
                "Ground Truth": ground_truth
            }
            return result
        except Exception as e:
            logging.error(f"Произошла ошибка при обработке вопроса {index + 1}: {e}")
            error_counter += 1
            time.sleep(0.5)  # Уменьшаем время ожидания между попытками
            continue
    return None


if __name__ == "__main__":
    logging.info("Начало выполнения 1_test_rag.py")
    current_time = get_current_time()

    with open('dataset/alfa_bank_dataset.json', "r", encoding="utf-8") as file:
        qa_dataset_fsk = json.load(file)["qa_dataset"]

    api_url = config["chat_rmr"]["api_url"]
    username = config["chat_rmr"]["username"]
    password = config["chat_rmr"]["password"]
    chat_id = config["chat_id"]
    token = login(api_url, username, password)

    if token:
        clear_chat_info(api_url, chat_id, token)
        filename = f'./1_result_rag/result_{current_time}.json'
        ensure_directory_exists(filename)
        total_questions = len(qa_dataset_fsk)
        results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_question, api_url, chat_id, token, qa_pair, config, index)
                       for index, qa_pair in enumerate(qa_dataset_fsk)]

            for future in tqdm(as_completed(futures), total=total_questions, desc="Processing questions"):
                result = future.result()
                if result:
                    results.append(result)
                    # Сортируем результаты по номеру вопроса перед сохранением
                    results.sort(key=lambda x: x["Question Number"])
                    save_results_to_file(results, filename)
        logging.info("Завершение выполнения 1_test_rag.py")
    else:
        logging.error(f"Не удалось получить токен для пользователя {username}")