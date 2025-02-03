import json
import asyncio
import os

import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from itertools import zip_longest
import time


def print_json(data):
    """Форматированный вывод JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def grouper(iterable, n):
    """Группировка элементов последовательности по n штук"""
    args = [iter(iterable)] * n
    return zip_longest(*args)


class AsyncLlamaTestRunner:
    def __init__(self):
        self.test_data_file = 'test_data_sceintists.json'
        self.current_results = {
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
        }#self._load_results()
        self.url = "https://llama3gpu.neuraldeep.tech/v1/chat/completions"
        self.api_key = "0facc28cb78c880add202510c9be25f7c33757b19a7cfe7f165ba4bd0c0e2d58!1iaj293"
        self.batch_size = 4
        self.results_file = "../results/llama_test_scientists_results.jsonl"
        self.all_results = []
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

    def save_result(self, result: Dict):
        """Сохраняет результат в JSONL файл"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def print_summary(self, results: List[Dict]):
        """Выводит статистику тестирования"""
        total = len(results)
        success = sum(1 for r in results if r.get('success', False))
        failed = total - success
        avg_time = sum(r.get('generation_time', 0) for r in results) / total if total > 0 else 0

        print("\n=== Статистика тестирования ===")
        print(f"Всего тестов: {total}")
        print(f"Успешно: {success}")
        print(f"Ошибок: {failed}")
        if total > 0:
            print(f"Точность: {(success / total) * 100:.2f}%")
            print(f"Среднее время генерации: {avg_time:.2f}с")
        print(f"Результаты сохранены в: {self.results_file}")
        print("=" * 50)

    async def route_query(self, session: aiohttp.ClientSession, scientist: str) -> Tuple[
        Union[Dict, None], float, str, Optional[str]]:
        """Асинхронно определяет подходящий домен для запроса"""

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
        query = f"Предоставь информацию об ученом: {scientist}"
        system_message = "Вы - помощник, предоставляющий информацию о людях, особенно об ученых. Пожалуйста, предоставьте точную и краткую информацию в соответствии с запрошенной структурой."

        request_data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Query: {query}"}
            ],
            "model": "llama-3-8b-instruct-8k",
            "max_tokens": 1000,
            "temperature": 0.1,
            "guided_json": json.dumps(json_schema),
            "guided_decoding_backend": "lm-format-enforcer"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            start_time = time.time()
            async with session.post(self.url, json=request_data, headers=headers, ssl=False) as response:
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

    async def process_question(self, session: aiohttp.ClientSession, scientist: str):
        """Обрабатывает один вопрос"""
        print(f"\n{'=' * 80}")
        print(f"Обработка учёного: {scientist}")
        print('-' * 80)

        try:
            # Получаем ответ модели
            model_response, generation_time, raw_response, error = await self.route_query(session, scientist)

            print("\nОтвет модели (raw):")
            print_json(raw_response)
            print('-' * 80)

            success = False
            error_message = None
            if model_response:
                success, error_message = self._validate_json_structure(model_response)
            else:
                error_message = error if error else "Не получен ответ от API"

            # Сохраняем результат теста
            test_result = {
                'test_number': self.test_number,
                'scientist': scientist,
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'error': error_message,
                'generation_time': round(generation_time, 2),
                'response': model_response if model_response else None,
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

            if model_response:
                print("\nОбработанный ответ:")
                print(json.dumps(model_response, ensure_ascii=False, indent=2))
                print("-" * 50)

        except Exception as e:
            print(f"❌ Критическая ошибка: {str(e)}")

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

    async def process_batch(self, session: aiohttp.ClientSession, scientists: List[str]):
        """Обрабатывает пакет вопросов"""
        tasks = []
        for scientist in scientists:
            if scientist:  # Проверяем на None из-за zip_longest
                tasks.append(self.process_question(session, scientist))
        batch_results = await asyncio.gather(*tasks)
        self.all_results.extend(batch_results)
        return batch_results

    async def run_tests(self):
        """Запускает тестирование"""
        print("=== Начало тестирования ===")
        print(f"Результаты будут сохранены в: {self.results_file}")
        print("=" * 50)

        # Очищаем файл результатов
        open(self.results_file, 'w').close()

        scientists = self._load_test_data()
        if not scientists:
            print("Не удалось загрузить список ученых из файла")
            return

        async with aiohttp.ClientSession() as session:
            for batch in zip_longest(*[iter(scientists)] * self.batch_size):
                valid_batch = [scientist for scientist in batch if scientist is not None]
                await self.process_batch(session, valid_batch)


    def _validate_json_structure(self, model_response: Dict) -> Tuple[bool, Optional[str]]:
        """
                Проверяет структуру JSON на соответствие схеме
                Returns:
                    Tuple[успех, сообщение_об_ошибке]
        """
        required_fields = {'name', 'age', 'daysoflife', 'dayofdeath', 'occupation', 'skills'}

        # Проверка наличия всех полей
        missing_fields = required_fields - set(model_response.keys())
        if missing_fields:
            return False, f"Отсутствуют обязательные поля: {', '.join(missing_fields)}"

        # Проверка типов данных
        if not isinstance(model_response['skills'], list):
            return False, "Поле 'skills' должно быть списком"

        type_checks = {
            'name': str,
            'age': int,
            'daysoflife': int,
            'dayofdeath': int,
            'occupation': str
        }

        for field, expected_type in type_checks.items():
            value = model_response[field]
            if value is None:
                return False, f"Поле '{field}' не может быть null"
            if not isinstance(value, expected_type):
                actual_type = type(value).__name__
                return False, f"Поле '{field}' должно быть типа {expected_type.__name__}, получено {actual_type}"

        return True, None

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
            f.write(json.dumps(self.current_results, ensure_ascii=False) + '\n')

    def _format_timestamp(self) -> str:
        """Форматирует текущее время"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


async def main():
    runner = AsyncLlamaTestRunner()
    await runner.run_tests()


if __name__ == "__main__":
    asyncio.run(main())