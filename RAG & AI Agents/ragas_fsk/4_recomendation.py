import json
import openai
import yaml
import os
from datetime import datetime
from tqdm import tqdm

# Загрузка конфигурации из файла config.yaml
with open("Новые результаты/config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

api_key = config["api_key"]
base_url = config["base_url"]

# Создание клиента OpenAI с кастомным хостом и API ключом
client = openai.OpenAI(api_key=api_key, base_url=base_url)

# Функция для отправки данных в агента и получения рекомендаций
def validate_response_with_llm(prompt):
    temperature = 0.0
    max_tokens = 300

    response = client.chat.completions.create(
        model="llama-3-8b-instruct-8k",  # Используем вашу модель
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Your answer must be in Russian."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    response_text = response.choices[0].message.content
    return response_text

# Функция для поиска самого последнего файла в папке
def find_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# Основная функция для обработки данных
def process_data():
    input_directory = './4_low_scores_result'
    input_filename = find_latest_file(input_directory)

    # Прочитать данные из последнего файла low_scores.json
    with open(input_filename, 'r', encoding='utf-8') as file:
        low_scores_data = json.load(file)

    # Открываем файл для записи рекомендаций
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'./5_recomendation_result/recommendations_{current_time}.json'
    with open(output_filename, 'w', encoding='utf-8') as file:
        all_recommendations = []

        # Обработка каждого вопроса отдельно с использованием tqdm для прогресс-бара
        for item in tqdm(low_scores_data, desc="Processing data"):
            extracted_data = {
                "question_number": item["Question Number"],
                "question": item["Question"],
                "answer": item["Answer"],
                "ground_truth": item["Ground Truth"],
                "context": item["Context"],
                "answer_relevancy": item["Evaluation"]["Answer Relevancy"],
                "context_relevancy": item["Evaluation"]["Context Relevancy"]
            }

            # Формируем запрос для агента
            prompt = f"""
            The following data is part of a Retrieval-Augmented Generation (RAG) system. Please analyze the data and provide a brief explanation in Russian of what is wrong with the answer. Indicate whether the answer does not correspond to the ground truth or the context, and suggest improving the answer by either replacing the ground truth or improving the context. Include the scores for Answer Relevancy and Context Relevancy.

            Data: {json.dumps(extracted_data, ensure_ascii=False, indent=4)}

            Please return the result in JSON format with the keys "question_number" and "recommendation".
            "question_number":
            "recommendation": 
             "text":    
            You answer must be in russian language
            """

            # Получить рекомендации от агента
            recommendations = validate_response_with_llm(prompt)

            # Проверка, является ли ответ корректным JSON
            try:
                recommendations_json = json.loads(recommendations)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                print("Response text was:", recommendations)
                continue

            # Добавляем дополнительные данные в результат
            recommendations_json["question"] = item["Question"]
            recommendations_json["answer"] = item["Answer"]
            recommendations_json["ground_truth"] = item["Ground Truth"]
            recommendations_json["context"] = item["Context"]

            # Добавляем рекомендации в общий список
            all_recommendations.append(recommendations_json)

            # Записываем текущие рекомендации в файл
            file.seek(0)
            json.dump(all_recommendations, file, ensure_ascii=False, indent=4)
            file.truncate()

if __name__ == "__main__":
    print("4_recomendation.py: Начало выполнения")
    process_data()
    print("4_recomendation.py: Завершение выполнения")