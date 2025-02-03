import asyncio
import aiohttp
import json
import yaml
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from itertools import zip_longest
import urllib3
import os
import requests

# Отключаем предупреждения о небезопасном SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def print_json(data):
    """Форматированный вывод JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def grouper(iterable, n):
    """Группировка элементов последовательности по n штук"""
    args = [iter(iterable)] * n
    return zip_longest(*args)


class TokenManager:
    def __init__(self):
        self.token_path = 'token.json'
        self.config_path = 'config.yaml'
        self.config = self._load_config()

    def _load_config(self):
        """Загружает конфигурацию из YAML файла"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
            return None

    def _get_new_token(self):
        """Получает новый токен через API"""
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
        """Сохраняет токен в файл"""
        try:
            with open(self.token_path, 'w') as file:
                json.dump({
                    'token': token_data['access_token'],
                    'created_at': datetime.now().timestamp()
                }, file)
        except Exception as e:
            print(f"Ошибка при сохранении токена: {e}")

    def _load_token(self):
        """Загружает существующий токен из файла"""
        try:
            if os.path.exists(self.token_path):
                with open(self.token_path, 'r') as file:
                    return json.load(file)
            print("Токен не найден, получаем новый")
            return None
        except Exception as e:
            print(f"Ошибка при загрузке токена: {e}")
            return None

    def _is_token_expired(self, token_data):
        """Проверяет, истек ли срок действия токена"""
        if not token_data:
            return True
        now = datetime.now().timestamp()
        created_at = token_data.get('created_at', 0)
        return (now - created_at) >= 1800  # 30 минут

    def get_auth_token(self):
        """Получает действующий токен"""
        token_data = self._load_token()

        if self._is_token_expired(token_data):
            print("Получение нового токена...")
            token_data = self._get_new_token()
            if not token_data:
                return None
            return token_data['access_token']

        return token_data['token']


class AsyncGigaChatClient:
    def __init__(self, token: str):
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.domains_info = [
            {"id": 1, "name": "ЖК 1 'Донской'", "chat_id": "6704d451dae88f75d8ae9986"},
            {"id": 2, "name": "ЖК 1 'Измайловский'", "chat_id": "6704d45edae88f75d8ae9988"},
            {"id": 3, "name": "ЖК 1 'Ленинградский'", "chat_id": "6704d46cdae88f75d8ae998a"},
            {"id": 4, "name": "ЖК 1 'Южный'", "chat_id": "6704d478dae88f75d8ae998c"},
            {"id": 5, "name": "ЖК 'Amber city'", "chat_id": "6704d48adae88f75d8ae998e"},
            {"id": 6, "name": "ЖК 'Южная Битца'", "chat_id": "6704d497dae88f75d8ae9990"}
        ]
        self.batch_size = 4
        self.results_file = '../results/gigachat_test_results.jsonl'

    async def route_query(self, session: aiohttp.ClientSession, query: str) -> Tuple[
        Union[Dict, None], float, str, Optional[str]]:
        """Асинхронно определяет подходящий домен для запроса"""
        domains_list = ", ".join([f"{d['name']} (chat_id: {d['chat_id']})" for d in self.domains_info])

        system_prompt = f"""You are a routing agent that determines the most appropriate residential complex for a query.
        Available complexes: {domains_list}

        Analyze the query and respond with:
        1. The selected complex's chat_id
        2. Your reasoning for why this complex is most appropriate
        3. The name of the selected complex

        You must respond with valid JSON containing EXACTLY these fields:
        {{
            "chat_id": "selected chat_id from the list",
            "reasoning": "detailed explanation in Russian for why this complex was chosen",
            "selected_complex": "exact name of the selected complex from the list"
        }}

        Do not add any text outside the JSON object. Ensure all fields are present and properly formatted."""

        payload = {
            "model": "GigaChat-Pro",
            "temperature": 0.1,
            "top_p": 0.47,
            "n": 1,
            "max_tokens": 2000,
            "repetition_penalty": 1.07,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ]
        }

        try:
            start_time = time.time()
            async with session.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    ssl=False
            ) as response:
                result = await response.json()
                raw_content = result['choices'][0]['message']['content']
                generation_time = time.time() - start_time

                try:
                    parsed_json = json.loads(raw_content)
                    return parsed_json, generation_time, raw_content, None
                except json.JSONDecodeError as e:
                    return None, generation_time, raw_content, f"Ошибка парсинга JSON: {str(e)}"

        except Exception as e:
            return None, 0.0, str(e), f"Ошибка API: {str(e)}"

    def save_result(self, result: Dict):
        """Сохраняет результат в JSONL файл"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    async def process_question(self, session: aiohttp.ClientSession, question: str, expected_domain: str):
        """Обрабатывает один вопрос"""
        print(f"\n{'=' * 80}")
        print(f"Обработка вопроса: {question}")
        print(f"Ожидаемый домен: {expected_domain}")
        print('-' * 80)

        try:
            # Получаем ответ модели
            model_response, generation_time, raw_response, error = await self.route_query(session, question)

            print("\nОтвет модели (raw):")
            print_json(raw_response)
            print('-' * 80)

            if model_response and isinstance(model_response, dict):
                # Проверяем соответствие выбранного домена ожидаемому
                domain_name = next((d['name'] for d in self.domains_info
                                    if d['chat_id'] == model_response.get('chat_id')), None)
                success = domain_name == expected_domain

                test_result = {
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'expected_domain': expected_domain,
                    'model_response': model_response,
                    'raw_response': raw_response,
                    'success': success,
                    'generation_time': round(generation_time, 2)
                }

                # Сохраняем результат
                self.save_result(test_result)

                # Вывод результата
                print("\nРезультат обработки:")
                print(f"{'✅' if success else '❌'} Статус: {'Успешно' if success else 'Ошибка'}")
                print(f"Выбранный ЖК: {model_response['selected_complex']}")
                print(f"Обоснование: {model_response['reasoning']}")
                print(f"Время генерации: {round(generation_time, 2)}с")
                print('-' * 80)

                return test_result
            else:
                print(f"❌ Ошибка: {error}")
                return {'success': False, 'error': error}

        except Exception as e:
            print(f"❌ Ошибка при обработке вопроса: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def process_batch(self, session: aiohttp.ClientSession, questions: List[tuple]):
        """Обрабатывает пакет вопросов"""
        tasks = []
        for question, domain in questions:
            if question:  # Проверяем на None из-за zip_longest
                tasks.append(self.process_question(session, question, domain))
        return await asyncio.gather(*tasks)

    def load_test_data(self) -> List[Dict]:
        """Загружает тестовые вопросы"""
        with open('test_data_domain_detemination.json', 'r', encoding='utf-8') as f:
            return json.load(f).get('test_questions', [])

    def print_summary(self, results: List[Dict]):
        """Выводит статистику тестирования"""
        total = len(results)
        success = sum(1 for r in results if r.get('success', False))
        failed = total - success

        print("\n=== Статистика тестирования ===")
        print(f"Всего тестов: {total}")
        print(f"Успешно: {success}")
        print(f"Ошибок: {failed}")
        if total > 0:
            print(f"Точность: {(success / total) * 100:.2f}%")
        print(f"Результаты сохранены в: {self.results_file}")
        print("=" * 50)

    async def run_tests(self):
        """Запускает тестирование"""
        print("=== Начало тестирования ===")
        print(f"Результаты будут сохранены в: {self.results_file}")
        print("=" * 50)

        # Очищаем файл результатов
        open(self.results_file, 'w').close()

        test_data = self.load_test_data()
        all_questions = [(q, data['domain'])
                         for data in test_data
                         for q in data['questions']]

        all_results = []

        async with aiohttp.ClientSession() as session:
            for batch in grouper(all_questions, self.batch_size):
                valid_batch = [(q, d) for q, d in batch if q is not None]
                batch_results = await self.process_batch(session, valid_batch)
                all_results.extend(batch_results)
                self.print_summary(all_results)

        self.print_summary(all_results)


async def main():
    token_manager = TokenManager()
    token = token_manager.get_auth_token()

    if not token:
        print("Не удалось получить токен")
        return

    client = AsyncGigaChatClient(token)
    await client.run_tests()


if __name__ == "__main__":
    asyncio.run(main())