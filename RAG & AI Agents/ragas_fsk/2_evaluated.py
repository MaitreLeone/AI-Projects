import json
import openai
import logging
import os
import yaml
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Настройка логирования
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Отключение вывода HTTP-запросов
logging.getLogger("openai").setLevel(logging.WARNING)

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Загрузка конфигурации из файла config.yaml
with open("config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Настройка кастомного хоста и модели
api_key = config["api_key"]
base_url = config["base_url"]

# Создание клиента OpenAI с кастомным хостом и API ключом
client = openai.OpenAI(api_key=api_key, base_url=base_url)

def validate_response_with_llm(system_prompt, user_prompt):
    temperature = 0.1
    max_tokens = 1

    response = client.chat.completions.create(
        model="llama-3-8b-instruct-8k",  # Используем вашу модель
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    time.sleep(3)
    response_text = response.choices[0].message.content.strip()
    response_text = response_text.replace('"', '').replace("'", "")  # Удаляем кавычки
    return float(response_text)  # Преобразуем строку в число

def evaluate_answer(question, answer, context, ground_truth):
    # Промпты для оценки
    answer_relevancy_system_prompt = """
    You are a Q&A system evaluator. 
    Your task is to evaluate the quality of the answer to the question provided based on relevance, completeness, and clarity. 
    Rate as 5 if the system fully answered or provided the necessary information, 
    and 0 if the system did not cope with the question. 
    Return only number the format: "X", where X is your evaluation.
    """
    answer_relevancy_user_prompt = f"""
    Question: {question}
    Answer: {answer}
    Context: {context}
    Possible example: {ground_truth}
    """

    context_relevancy_system_prompt = """
    You compare the question and the context. 
    Your task is to evaluate whether the context can be used to answer the question. 
    Rate as 5 if the context is suitable or even partially suitable, 
    and 0  if the context is not suitable at all. 
    Please return only number of your rating only in the following format: 'X', where X is your rating.
    """
    context_relevancy_user_prompt = f"""
    Question: {question}
    Answer: {answer}
    Context: {context}
    """

    # Получение оценок
    answer_relevancy_score = validate_response_with_llm(answer_relevancy_system_prompt, answer_relevancy_user_prompt)
    context_relevancy_score = validate_response_with_llm(context_relevancy_system_prompt, context_relevancy_user_prompt)

    return {
        "Answer Relevancy": answer_relevancy_score,
        "Context Relevancy": context_relevancy_score
    }

def analyze_results(input_filename, output_filename):
    with open(input_filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    total_answer_relevancy = 0
    total_context_relevancy = 0
    count = 0
    max_score = 5  # Максимальная оценка

    def process_item(item):
        question = item["Question"]
        answer = item["Answer"]
        context = item["Context"]
        ground_truth = item["Ground Truth"]

        # Оценка ответа
        evaluation = evaluate_answer(question, answer, context, ground_truth)

        item["Evaluation"] = evaluation
        return item

    def print_progress():
        while not stop_thread:
            if count > 0:
                average_answer_relevancy_percentage = (total_answer_relevancy / (count * max_score)) * 100
                average_context_relevancy_percentage = (total_context_relevancy / (count * max_score)) * 100
                # Вывод в терминал
                print(f"Average Answer Relevancy: {average_answer_relevancy_percentage:.2f}%")
                print(f"Average Context Relevancy: {average_context_relevancy_percentage:.2f}%")
            time.sleep(5)

    stop_thread = True
    progress_thread = threading.Thread(target=print_progress)
    progress_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in data}
            for future in tqdm(as_completed(future_to_item), total=len(data), desc="Evaluating answers", mininterval=1):
                item = future_to_item[future]
                try:
                    result = future.result()
                    evaluation = result["Evaluation"]

                    # Обновление среднего скора
                    total_answer_relevancy += evaluation["Answer Relevancy"]
                    total_context_relevancy += evaluation["Context Relevancy"]
                    count += 1

                    average_answer_relevancy = total_answer_relevancy / count
                    average_context_relevancy = total_context_relevancy / count

                    result["Average Answer Relevancy"] = average_answer_relevancy
                    result["Average Context Relevancy"] = average_context_relevancy

                    results.append(result)

                    # Сортировка результатов по номеру вопроса
                    results.sort(key=lambda x: x["Question Number"])

                    # Сохранение результатов в файл после каждой итерации
                    with open(output_filename, "w", encoding="utf-8") as file:
                        json.dump(results, file, ensure_ascii=False, indent=4)

                except Exception as e:
                    logging.error(f"An error occurred while processing item {item['Question Number']}: {e}")
    finally:
        stop_thread = True
        progress_thread.join()

def find_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    latest_file = max(files, key=os.path.getctime)
    return latest_file

if __name__ == "__main__":
    logging.info("Начало выполнения 2_evaluated.py")
    input_directory = '1_result_rag'
    input_filename = find_latest_file(input_directory)
    current_time = get_current_time()
    output_filename = f'./2_evaluated_result/evaluated_results{current_time}.json'
    analyze_results(input_filename, output_filename)
    print("Script execution completed.")
    logging.info("Завершение выполнения 2_evaluated.py")