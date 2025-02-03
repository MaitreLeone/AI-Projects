import requests
import urllib3
import json
import yaml
import os
import uuid
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Отключаем предупреждения о небезопасном SSL-соединении
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class TokenManager:
    def __init__(self):
        self.token_path = 'token.json'
        self.config_path = 'config.yaml'
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
            return None

    def _get_new_token(self):
        if not self.config:
            print("Ошибка: отсутствует конфигурация")
            return None

        url = self.config['GCHAT_AUTH_URL']
        headers = {
            "Authorization": f"Basic {self.config['GCHAT_AUTHORIZATION_DATA']}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "scope": self.config['GCHAT_SCOPE'].replace("scope=", "")
        }

        try:
            response = requests.post(url, headers=headers, data=data, verify=False)
            response.raise_for_status()
            token_data = response.json()
            self._save_token(token_data)
            return token_data
        except Exception as e:
            print(f"Ошибка при получении нового токена: {e}")
            return None

    def _save_token(self, token_data):
        try:
            with open(self.token_path, 'w') as file:
                json.dump({
                    'token': token_data['access_token'],
                    'created_at': datetime.now().timestamp()
                }, file)
        except Exception as e:
            print(f"Ошибка при сохранении токена: {e}")

    def _load_token(self):
        try:
            if os.path.exists(self.token_path):
                with open(self.token_path, 'r') as file:
                    return json.load(file)
            return None
        except Exception as e:
            print(f"Токен не найден, получаем новый")
            return None

    def _is_token_expired(self, token_data):
        if not token_data:
            return True
        now = datetime.now().timestamp()
        created_at = token_data.get('created_at', 0)
        return (now - created_at) >= 1800

    def get_auth_token(self):
        token_data = self._load_token()

        if self._is_token_expired(token_data):
            print("Получение нового токена...")
            token_data = self._get_new_token()
            if not token_data:
                return None
            return token_data['access_token']

        return token_data['token']


class GigaChatClient:
    def __init__(self, token):
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def generate_json(self, user_query: str, test_number: int) -> Tuple[Union[Dict, None], float, str, Optional[str]]:
        """
        Генерирует JSON с информацией об ученом
        Args:
            user_query: Запрос к API
            test_number: Номер текущего теста
        Returns:
            Tuple[результат_json, время_генерации, сырой_ответ, ошибка]
        """
        system_prompt = (
            'Сгенерируй информацию о человеке в формате JSON. '
            'Придерживайся следующих правил: все поля должны быть заполнены корректными данными, '
            'не придумывай несуществующие данные. '
            'Результат верни в формате JSON без каких-либо пояснений, например: '
            '{"name": "имя человека", "age": число_лет, "daysoflife": количество_дней_жизни, '
            '"dayofdeath": "дата смерти или - если человек жив", "occupation": "профессия", '
            '"skills": ["навык1", "навык2"]}. '
            'Поле dayofdeath должно быть строкой, если человек жив, укажи "-"'
        )

        payload = {
            "model": "GigaChat-Pro",
            "temperature": 0.1,
            "top_p": 0.47,
            "n": 1,
            "max_tokens": 512,
            "repetition_penalty": 1.07,
            "stream": False,
            "update_interval": 0,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Test #{test_number}. {user_query}"
                }
            ]
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                verify=False
            )
            response.raise_for_status()
            result = response.json()
            raw_content = result['choices'][0]['message']['content']
            generation_time = time.time() - start_time

            try:
                parsed_json = json.loads(raw_content)
                return parsed_json, generation_time, raw_content, None
            except json.JSONDecodeError as e:
                return None, generation_time, raw_content, f"Ошибка парсинга JSON: {str(e)}"

        except Exception as e:
            return None, 0.0, str(e), f"Ошибка API: {str(e)}"


class TestRunner:
    def __init__(self, client: GigaChatClient):
        self.client = client
        self.test_data_file = '../llama-3-1-lm-format-enforcer/test_data_sceintists.json'
        self.results_file = '../results/gigachat_test_scientists_results.json'
        self.current_results = self._load_results()
        self.test_number = 0

    def _load_test_data(self) -> List[str]:
        """Загружает список ученых из JSON файла"""
        try:
            with open(self.test_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('scientists', [])
        except Exception as e:
            print(f"Ошибка при загрузке тестовых данных: {str(e)}")
            return []

    def _load_results(self) -> Dict:
        """Загружает существующие результаты тестов"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                self.test_number = results.get('summary', {}).get('total', 0)
                return results
        return {
            'tests': [],
            'summary': {
                'total': 0,
                'success': 0,
                'failed': 0,
                'total_generation_time': 0.0,
                'average_generation_time': 0.0,
                'start_time': datetime.now().isoformat(),
                'end_time': None
            }
        }

    def _save_results(self):
        """Сохраняет результаты тестов"""
        self.current_results['summary']['end_time'] = datetime.now().isoformat()
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_results, f, ensure_ascii=False, indent=2)

    def _validate_json_structure(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Проверяет структуру JSON на соответствие схеме
        Returns:
            Tuple[успех, сообщение_об_ошибке]
        """
        required_fields = {'name', 'age', 'daysoflife', 'dayofdeath', 'occupation', 'skills'}

        # Проверка наличия всех полей
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            return False, f"Отсутствуют обязательные поля: {', '.join(missing_fields)}"

        # Проверка типов данных
        if not isinstance(data['skills'], list):
            return False, "Поле 'skills' должно быть списком"

        type_checks = {
            'name': str,
            'age': int,
            'daysoflife': int,
            'dayofdeath': str,
            'occupation': str
        }

        for field, expected_type in type_checks.items():
            value = data[field]
            if value is None:
                return False, f"Поле '{field}' не может быть null"
            if not isinstance(value, expected_type):
                actual_type = type(value).__name__
                return False, f"Поле '{field}' должно быть типа {expected_type.__name__}, получено {actual_type}"

        return True, None

    def _format_timestamp(self) -> str:
        """Форматирует текущее время"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def run_tests(self):
        """Запускает тесты для всех ученых из файла"""
        print("\n=== Начало тестирования ===")
        print(f"Время начала: {self._format_timestamp()}")
        print("-" * 50)

        scientists = self._load_test_data()
        if not scientists:
            print("Не удалось загрузить список ученых из файла")
            return

        for scientist in scientists:
            self.test_number += 1
            print(f"\nТест #{self.test_number} | Ученый: {scientist}")
            print("-" * 30)

            try:
                # Генерируем запрос
                query = f"Предоставь информацию об ученом: {scientist}"
                result, generation_time, raw_response, error_details = self.client.generate_json(
                    query,
                    self.test_number
                )

                # Проверяем результат
                success = False
                error_message = None

                if result:
                    success, error_message = self._validate_json_structure(result)
                else:
                    error_message = error_details if error_details else "Не получен ответ от API"

                # Сохраняем результат теста
                test_result = {
                    'test_number': self.test_number,
                    'scientist': scientist,
                    'timestamp': datetime.now().isoformat(),
                    'success': success,
                    'error': error_message,
                    'generation_time': round(generation_time, 2),
                    'response': result if result else None,
                    'raw_response': raw_response
                }

                self.current_results['tests'].append(test_result)

                # Обновляем статистику
                self.current_results['summary']['total'] += 1
                self.current_results['summary']['total_generation_time'] += generation_time
                if success:
                    self.current_results['summary']['success'] += 1
                else:
                    self.current_results['summary']['failed'] += 1

                # Обновляем среднее время генерации
                self.current_results['summary']['average_generation_time'] = round(
                    self.current_results['summary']['total_generation_time'] /
                    self.current_results['summary']['total'],
                    2
                )

                # Сохраняем результаты после каждого теста
                self._save_results()

                # Выводим результат теста
                status = "✅ Успешно" if success else f"❌ Ошибка: {error_message}"
                print(f"Статус: {status}")
                print(f"Время генерации: {round(generation_time, 2)}с")
                print("\nСырой ответ API:")
                print("-" * 50)
                print(raw_response)
                print("-" * 50)

                if result:
                    print("\nОбработанный ответ:")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    print("-" * 50)

            except Exception as e:
                print(f"❌ Критическая ошибка: {str(e)}")

        # Выводим итоговую статистику
        summary = self.current_results['summary']
        print("\n=== Итоговые результаты тестирования ===")
        print(f"Время окончания: {self._format_timestamp()}")
        print("-" * 50)
        print(f"Всего тестов: {summary['total']}")
        print(f"Успешно: {summary['success']}")
        print(f"Ошибок: {summary['failed']}")
        print(f"Процент успеха: {(summary['success'] / summary['total'] * 100):.2f}%")
        print(f"Среднее время генерации: {summary['average_generation_time']}с")
        print(f"Общее время генерации: {round(summary['total_generation_time'], 2)}с")
        print("=" * 50)
        print(f"Результаты сохранены в файл: {self.results_file}")
        print("=" * 50)


def main():
    # Инициализация и запуск тестов
    token_manager = TokenManager()
    token = token_manager.get_auth_token()

    if not token:
        print("Не удалось получить токен")
        return

    client = GigaChatClient(token)
    test_runner = TestRunner(client)
    test_runner.run_tests()


if __name__ == "__main__":
    main()