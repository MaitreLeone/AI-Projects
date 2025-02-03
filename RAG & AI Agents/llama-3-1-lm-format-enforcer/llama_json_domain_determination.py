import json
import asyncio
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
        self.url = "https://llama3gpu.neuraldeep.tech/v1/chat/completions"
        self.api_key = "0facc28cb78c880add202510c9be25f7c33757b19a7cfe7f165ba4bd0c0e2d58!1iaj293"
        self.batch_size = 4
        self.results_file = '../results/llama_test_results.jsonl'
        self.domains = [
            {"id": 1, "name": "ЖК 1 'Донской'", "chat_id": "6704d451dae88f75d8ae9986"},
            {"id": 2, "name": "ЖК 1 'Измайловский'", "chat_id": "6704d45edae88f75d8ae9988"},
            {"id": 3, "name": "ЖК 1 'Ленинградский'", "chat_id": "6704d46cdae88f75d8ae998a"},
            {"id": 4, "name": "ЖК 1 'Южный'", "chat_id": "6704d478dae88f75d8ae998c"},
            {"id": 5, "name": "ЖК 'Amber city'", "chat_id": "6704d48adae88f75d8ae998e"},
            {"id": 6, "name": "ЖК 'Южная Битца'", "chat_id": "6704d497dae88f75d8ae9990"}
        ]
        self.all_results = []

    def load_questions(self) -> List[Dict]:
        """Загружает тестовые вопросы"""
        with open('test_data_domain_detemination.json', 'r', encoding='utf-8') as f:
            return json.load(f).get('test_questions', [])

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

    async def route_query(self, session: aiohttp.ClientSession, query: str) -> Tuple[
        Union[Dict, None], float, str, Optional[str]]:
        """Асинхронно определяет подходящий домен для запроса"""
        domains_list = ", ".join([f"{d['name']} (chat_id: {d['chat_id']})" for d in self.domains])

        json_schema = {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string"},
                "reasoning": {"type": "string"},
                "selected_complex": {"type": "string"}
            },
            "required": ["chat_id", "reasoning", "selected_complex"]
        }

        system_message = f"""You are a routing agent that determines the most appropriate residential complex for a query.
        Available complexes: {domains_list}

        Analyze the query and respond with:
        1. The selected complex's chat_id
        2. Your reasoning for why this complex is most appropriate
        3. The name of the selected complex

        You must respond with valid JSON containing chat_id, reasoning, and selected_complex fields."""

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
                domain_name = next((d['name'] for d in self.domains
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
                error_result = {
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'expected_domain': expected_domain,
                    'error': error,
                    'success': False,
                    'generation_time': 0.0
                }
                self.save_result(error_result)
                print(f"❌ Ошибка: {error}")
                return error_result

        except Exception as e:
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'expected_domain': expected_domain,
                'error': str(e),
                'success': False,
                'generation_time': 0.0
            }
            self.save_result(error_result)
            print(f"❌ Ошибка при обработке вопроса: {str(e)}")
            return error_result

    async def process_batch(self, session: aiohttp.ClientSession, questions: List[tuple]):
        """Обрабатывает пакет вопросов"""
        tasks = []
        for question, domain in questions:
            if question:  # Проверяем на None из-за zip_longest
                tasks.append(self.process_question(session, question, domain))
        batch_results = await asyncio.gather(*tasks)
        self.all_results.extend(batch_results)
        self.print_summary(self.all_results)
        return batch_results

    async def run_tests(self):
        """Запускает тестирование"""
        print("=== Начало тестирования ===")
        print(f"Результаты будут сохранены в: {self.results_file}")
        print("=" * 50)

        # Очищаем файл результатов
        open(self.results_file, 'w').close()

        test_data = self.load_questions()
        all_questions = [(q, data['domain'])
                         for data in test_data
                         for q in data['questions']]

        async with aiohttp.ClientSession() as session:
            for batch in zip_longest(*[iter(all_questions)] * self.batch_size):
                valid_batch = [(q, d) for q, d in batch if q is not None]
                await self.process_batch(session, valid_batch)

        self.print_summary(self.all_results)


async def main():
    runner = AsyncLlamaTestRunner()
    await runner.run_tests()


if __name__ == "__main__":
    asyncio.run(main())