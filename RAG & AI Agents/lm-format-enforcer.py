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
        "occupation": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age", "occupation", "skills"]
}

# URL вашего vLLM сервера
url = "http://62.68.146.188//v1/chat/completions"

# Получение пользовательского промпта
user_prompt = input("Введите ваш запрос о каком-либо ученом: ")

# Подготовка запроса
request_data = {
    "messages": [
        {"role": "system", "content": "Вы - помощник, предоставляющий информацию о людях, особенно об ученых. Пожалуйста, предоставьте точную и краткую информацию в соответствии с запрошенной структурой."},
        {"role": "user", "content": user_prompt}
    ],
    "model": "llama-3.1-8b-instruct",
    "max_tokens": 200,
    "temperature": 0.1,
    "guided_json": json.dumps(json_schema),
    "guided_decoding_backend": "lm-format-enforcer",
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "top_p": 1,
    "n": 1,
    "stream": False
}

# Отправка запроса
response = requests.post(url, json=request_data)

# Проверка успешности запроса
if response.status_code == 200:
    result = response.json()
    content = json.loads(result['choices'][0]['message']['content'])
    print("\nПолученная информация:")
    print_json(content)
else:
    print(f"Ошибка: {response.status_code}")
    print_json(response.json())
