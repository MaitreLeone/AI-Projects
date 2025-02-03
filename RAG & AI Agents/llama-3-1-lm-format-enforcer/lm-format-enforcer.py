import json
import requests

def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))

# Определение JSON-схемы для ответа
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "daysoflife": {"type": "integer"},
        "dayofdeath": {"type": "integer"},
        "occupation": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "daysoflife", "dayofdeath", "occupation", "skills"]
}

# Обновленный URL вашего vLLM сервера
url = "https://llama3gpu.neuraldeep.tech/v1/chat/completions"

# API ключ
api_key = "0facc28cb78c880add202510c9be25f7c33757b19a7cfe7f165ba4bd0c0e2d58!1iaj293"

# Получение пользовательского промпта
user_prompt = input("Введите ваш запрос о каком-либо ученом: ")

# Подготовка запроса
request_data = {
    "messages": [
        {"role": "system", "content": "Вы - помощник, предоставляющий информацию о людях, особенно об ученых. Пожалуйста, предоставьте точную и краткую информацию в соответствии с запрошенной структурой."},
        {"role": "user", "content": user_prompt}
    ],
    "model": "llama-3-8b-instruct-8k",
    "max_tokens": 1000,
    "temperature": 0.0,
    "guided_json": json.dumps(json_schema),
    "guided_decoding_backend": "lm-format-enforcer"
}

# Заголовки с авторизацией
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Отправка запроса
response = requests.post(url, json=request_data, headers=headers)

# Проверка успешности запроса
if response.status_code == 200:
    result = response.json()
    content = json.loads(result['choices'][0]['message']['content'])
    print("\nПолученная информация:")
    print_json(content)
else:
    print(f"Ошибка: {response.status_code}")
    print_json(response.json())