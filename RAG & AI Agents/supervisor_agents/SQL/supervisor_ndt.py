import psycopg2
import requests
import json
import base64
import yaml
from typing import Dict, Any, List
import logging
import sys
from psycopg2.extras import RealDictCursor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Загружаем конфигурацию из файла
with open("../config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Полная конфигурация для meta_override
meta_override_config = {
    "return_debug_data": True,
    "gpt_aggregation_prompt": "I am going to ask you a question, which I would like you to answer based only on the provided context and conversation history and not any other information. If there is not enough information in the context to answer the question, say 'I am not sure', then try to make a guess. Break your answer up into nicely readable paragraphs. Answer should be in Russian language",
    "model": "NDT_LLM_4",  # llama3.1
    "temperature": 0.1,
    "context_entries_num": 30,
    "history_entries_num": 0,
    "reranking": {
        "use": True,
        "model": "NDT_RR_1",
        "return_entries_num": 5
    },
    "guardrail_in": {
        "use": False,
        "stop_words": [
            "string"
        ]
    },
    "guardrail_out": {
        "use": False,
        "reranker_average_score": 0.2,
        "prompt_system_message": "prompt_system"
    }
}

# Информация о доменах и чатах
domains_info = [
    {"id": 1, "name": "ЖК 1 'Донской'", "chat_id": "6704d451dae88f75d8ae9986"},
    {"id": 2, "name": "ЖК 1 'Измайловский'", "chat_id": "6704d45edae88f75d8ae9988"},
    {"id": 3, "name": "ЖК 1 'Ленинградский'", "chat_id": "6704d46cdae88f75d8ae998a"},
    {"id": 4, "name": "ЖК 1 'Южный'", "chat_id": "6704d478dae88f75d8ae998c"},
    {"id": 5, "name": "ЖК 'Amber city'", "chat_id": "6704d48adae88f75d8ae998e"},
    {"id": 6, "name": "ЖК 'Южная Битца'", "chat_id": "6704d497dae88f75d8ae9990"}
]

DB_PARAMS = {
            "dbname": "fsk",
            "user": "admin",
            "password": "Lol770905",
            "host": "62.68.146.188",
            "port": "5432"
        }

VLLM_URL = "http://62.68.146.188/v1/chat/completions"

DB_SCHEMA = """
Tables in the database:

1. residential_complexes
   - id (SERIAL PRIMARY KEY)
   - name (TEXT UNIQUE)
   - chat_id (TEXT UNIQUE)

2. flats
   - id (SERIAL PRIMARY KEY)
   - complex_id (INTEGER, FOREIGN KEY referencing residential_complexes(id))
   - size_sqm (INTEGER)
   - num_rooms (INTEGER)
   - comfort_level (TEXT)
   - window_size (TEXT)
   - avg_price_per_sqm (INTEGER)

3. parking
   - id (SERIAL PRIMARY KEY)
   - complex_id (INTEGER, FOREIGN KEY referencing residential_complexes(id))
   - total_spots (INTEGER)

Relationships:
- flats.complex_id references residential_complexes.id
- parking.complex_id references residential_complexes.id

Available residential complexes:
"""

class Agent:
    def __init__(self, name: str, capabilities: str):
        self.name = name
        self.capabilities = capabilities

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")


class RouterAgent(Agent):
    def __init__(self, name: str, capabilities: str, domains_info: List[Dict[str, str]], api_url: str, api_key: str):
        super().__init__(name, capabilities)
        self.domains_info = domains_info
        self.api_url = api_url
        self.api_key = api_key

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        query = task['query']

        domains_list = ", ".join([f'{d["name"]} (chat_id: {d["chat_id"]})' for d in self.domains_info])
        system_message = f"""You are a routing agent. Your task is to select the most appropriate knowledge domain for a given query.
        Available domains: {domains_list}.
        If no domain is clearly suitable, choose the chat_id for "Общая база знаний".
        Respond in JSON format with fields "chat_id" (selected chat_id) and "reasoning" (explanation for the choice)."""

        user_message = f"Запрос: {query}"

        schema = {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string"},
                "reasoning": {"type": "string"}
            },
            "required": ["chat_id", "reasoning"]
        }

        request_data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "model": "llama-3-8b-instruct-8k",
            "max_tokens": 2000,
            "temperature": 0.1,
            "guided_json": json.dumps(schema),
            "guided_decoding_backend": "lm-format-enforcer"
        }

        logger.info(f"RouterAgent: Отправка запроса к LLM: {json.dumps(request_data, ensure_ascii=False)}")

        response = requests.post(self.api_url, json=request_data, headers={'Authorization': f'Bearer {self.api_key}'})

        logger.info(f"RouterAgent: Получен ответ от LLM. Статус: {response.status_code}")
        logger.info(f"RouterAgent: Текст ответа: {response.text}")

        if response.status_code == 200:
            llm_response = json.loads(response.json()['choices'][0]['message']['content'])
            logger.info(f"RouterAgent: Обработанный ответ LLM: {json.dumps(llm_response, ensure_ascii=False)}")

            selected_domain = next(
                (domain for domain in self.domains_info if domain['chat_id'] == llm_response['chat_id']), None)

            if selected_domain:
                result = {
                    "domain_name": selected_domain["name"],
                    "domain_id": selected_domain["id"],
                    "chat_id": selected_domain["chat_id"],
                    "reasoning": llm_response["reasoning"]
                }
            else:
                default_domain = next(
                    domain for domain in self.domains_info if domain["name"] == "Общая база знаний")
                result = {
                    "domain_name": default_domain["name"],
                    "domain_id": default_domain["id"],
                    "chat_id": default_domain["chat_id"],
                    "reasoning": "Выбран домен по умолчанию, так как указанный chat_id не найден."
                }

            logger.info(f"RouterAgent: Результат обработки: {json.dumps(result, ensure_ascii=False)}")
            return result
        else:
            logger.error(f"RouterAgent: Ошибка при отправке запроса к LLM. Код ответа: {response.status_code}")
            logger.error(f"RouterAgent: Текст ответа: {response.text}")

            default_domain = next(
                domain for domain in self.domains_info if domain["name"] == "Общая база знаний")
            result = {
                "name": default_domain["name"],
                "id": default_domain["id"],
                "chat_id": default_domain["chat_id"],
                "reasoning": "Выбран домен по умолчанию из-за ошибки в процессе выбора."
            }
            logger.info(f"RouterAgent: Использован домен по умолчанию: {json.dumps(result, ensure_ascii=False)}")
            return result

class SQLAgent(Agent):
    def __init__(self, name: str, capabilities: str, api_url: str, username: str, password: str, db_schema: str):
        super().__init__(name, capabilities)
        self.api_url = api_url
        self.username = username
        self.password = password
        self.token = self.login()
        self.db_schema = db_schema

        self.db_schema += "\n".join(
            [f"- {complex['name']} (chat_id: {complex['chat_id']})" for complex in domains_info])

    def login(self):
        credentials = base64.b64encode(f"{self.username}:{self.password}".encode('utf-8')).decode('utf-8')
        logger.info(f"SQLAgent: Попытка авторизации для пользователя {self.username}")
        response = requests.post(f"{self.api_url}/sign_in",
                                 headers={'accept': 'application/json', 'Authorization': f'Basic {credentials}'})

        if response.status_code == 200 or response.status_code == 201:
            logger.info("SQLAgent: Авторизация успешна")
            return response.json().get("access_token")
        else:
            logger.error(f"SQLAgent: Ошибка авторизации. Код ответа: {response.status_code}")
            logger.error(f"SQLAgent: Текст ответа: {response.text}")
            return None

    def pre_check_query(self, user_prompt):
        system_prompt = f"""You are an AI assistant that determines whether a user's query requires generating an SQL query based on the following database schema:

        {self.db_schema}

        Analyze the user's input and determine if it requires an SQL query to be answered.
        Return true if an SQL query is needed, false otherwise.
        Provide a brief explanation for your decision.
        """

        # New schema for pre-check
        pre_check_schema = {
            "type": "object",
            "properties": {
                "generate_sql": {"type": "boolean"},
                "explanation": {"type": "string"}
            },
            "required": ["generate_sql", "explanation"]
        }

        request_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": "llama-3-8b-instruct-8k",
            "max_tokens": 2000,
            "temperature": 0.0,
            "guided_json": json.dumps(pre_check_schema),
            "guided_decoding_backend": "lm-format-enforcer"
        }

        response = requests.post(VLLM_URL, json=request_data)

        if response.status_code == 200:
            result = response.json()
            return json.loads(result["choices"][0]["message"]["content"])
        else:
            logger.error(f"Error in pre-check: {response.status_code}")
            return None

    def get_sql_query(self, user_prompt):
        system_prompt = f"""You are an AI assistant that translates natural language queries into SQL. 
        Use the following database schema information to generate accurate SQL queries:

        {self.db_schema}

        Generate SQL queries based on user natural language input.
        In the query field, return none if the query does not match the database schema
        Use parameterized queries with %s placeholders for security.
        Use chat_id to identify specific apartment complexes

        """

        # SQL query schema
        sql_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "params": {"type": "array", "items": {"type": "string"}},
                "explanation": {"type": "string"},
                "rejected": {"type": "boolean"}
            },
            "required": ["query", "params", "explanation", "rejected"]
        }

        request_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "model": "llama-3-8b-instruct-8k",
            "max_tokens": 200,
            "temperature": 0.0,
            "guided_json": json.dumps(sql_schema),
            "guided_decoding_backend": "lm-format-enforcer"
        }

        response = requests.post(VLLM_URL, json=request_data)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"Error: {response.status_code}")
            return None

    def execute_sql_query(self, chat_id: str, query: str, params: str):
        if not self.token:
            logger.error("SQLAgent: Ошибка: не удалось авторизоваться")
            return None

        url = f'{self.api_url}/search/{chat_id}'
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    if not results:
                        # Если результатов нет, проверим наличие данных
                        cur.execute("SELECT COUNT(*) FROM residential_complexes WHERE chat_id = %s", [params[0]])
                        complex_count = cur.fetchone()['count']
                        if complex_count == 0:
                            logger.info(f"Жилой комплекс с chat_id '{params[0]}' не найден в базе данных.")

                        if "parking" in query.lower():
                            cur.execute(
                                "SELECT COUNT(*) FROM parking JOIN residential_complexes ON parking.complex_id = residential_complexes.id WHERE residential_complexes.chat_id = %s",
                                [params[0]])
                            parking_count = cur.fetchone()['count']
                            if parking_count == 0:
                                logger.info(f"Информация о паркинге для жилого комплекса с chat_id '{params[0]}' отсутствует.")

                        logger.info("Запрос выполнен успешно, но не вернул результатов. Возможно, данные отсутствуют.")
                    return results
        except psycopg2.Error as e:
            return f"Произошла ошибка при выполнении запроса: {e}"

    def process(self, task: Dict[str, Any]):
        chat_id = task['chat_id']
        user_prompt = task['query']
        if user_prompt:
            pre_check_result = self.pre_check_query(user_prompt)
            if pre_check_result is None:
                logger.error("Ошибка при выполнении предварительной проверки.")
            else:
                logger.info("SQLAgent: Результат предварительной проверки:")
                logger.info(f"Требуется SQL-запрос: {'Да' if pre_check_result['generate_sql'] else 'Нет'}")
                logger.info(f"Объяснение: {pre_check_result['explanation']}")

                if not pre_check_result['generate_sql']:
                    logger.warning(
                        "Запрос не требует генерации SQL. Попробуйте переформулировать вопрос или задать другой.")
                else:
                    # Второй этап: генерация и выполнение SQL-запроса
                    logger.info("SQLAgent: Генерация SQL-запроса:")
                    agent_output = self.get_sql_query(user_prompt)

                    try:
                        sql_data = json.loads(agent_output)

                        logger.info(f"SQLAgent: Сгенерированный SQL-запрос: {sql_data['query']}")

                        logger.info(f"Параметры: {sql_data['params']}")
                        logger.info(f"Объяснение: {sql_data['explanation']}")

                        result = self.execute_sql_query(chat_id, sql_data['query'], sql_data['params'])
                        if isinstance(result, str):
                            logger.info(f"SQLAgent: Результат запроса: {result}")
                            return result
                        else:
                            logger.info(f"SQLAgent: Результат запроса: {json.dumps(result)}")
                            return json.dumps(result, ensure_ascii=False)
                    except json.JSONDecodeError:
                        logger.error("Не удалось разобрать вывод агента как JSON. Вывод может быть не в ожидаемом формате.")
                    except KeyError as e:
                        logger.error(f"В выводе агента отсутствует ожидаемый ключ: {e}")
        else:
            logger.warning("Пожалуйста, введите запрос перед выполнением.")


class ResponseGenerator(Agent):
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        sql_response = task['sql_response']
        logger.info(f"ResponseGenerator: Получен ответ от SQL: {json.dumps(sql_response, ensure_ascii=False)}")
        if sql_response:
            result = {"result": sql_response}
        else:
            result = {"result": "Извините, не удалось получить ответ."}
        logger.info(f"ResponseGenerator: Сгенерирован ответ: {json.dumps(result, ensure_ascii=False)}")
        return result


class Supervisor:
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.name: agent for agent in agents}

    def execute_task(self, task: str) -> Dict[str, Any]:
        try:
            logger.info(f"Supervisor: Начало выполнения задачи: {task}")

            # Шаг 1: Маршрутизация
            router_result = self.agents['RouterAgent'].process({"query": task})
            logger.info(f"Supervisor: Результат маршрутизации: {json.dumps(router_result, ensure_ascii=False)}")

            # Шаг 2: SQL запрос
            sql_result = self.agents['SQLAgent'].process({
                "chat_id": router_result['chat_id'],
                "query": task
            })
            logger.info(f"Supervisor: Результат SQL запроса: {json.dumps(sql_result, ensure_ascii=False)}")

            # Шаг 3: Генерация ответа
            response = self.agents['ResponseGenerator'].process({
                "sql_response": sql_result
            })
            logger.info(f"Supervisor: Сгенерированный ответ: {json.dumps(response, ensure_ascii=False)}")


            result = {
                "domain": router_result['domain_name'],
                "reasoning": router_result['reasoning'],
                "query": task,
                "result": response['result']
            }
            logger.info(f"Supervisor: Итоговый результат: {json.dumps(result, ensure_ascii=False)}")
            return result
        except Exception as e:
            logger.error(f"Supervisor: Ошибка при выполнении задачи: {str(e)}", exc_info=True)
            return {"error": str(e)}


# Инициализация агентов и супервизора
router_agent = RouterAgent("RouterAgent", "Route queries to appropriate knowledge domains", domains_info,
                           config['llm']['url'], config['llm']['api_key'])
sql_agent = SQLAgent("SQLAgent", "Execute SQL queries to retrieve precise information from the database",
                     config['api']['url'], config['api']['username'], config['api']['password'], db_schema=DB_SCHEMA)
response_generator = ResponseGenerator("ResponseGenerator", "Generate human-readable responses based on SQL output")

supervisor = Supervisor([router_agent, sql_agent, response_generator])

AGENT_DESCRIPTIONS = {
    "RouterAgent": {
        "name": "RouterAgent",
        "capabilities": "Route queries to appropriate knowledge domains",
        "description": "Выбирает подходящий домен знаний на основе запроса пользователя."
    },
    "SQLAgent": {
        "name": "SQLAgent",
        "capabilities": "Execute SQL queries to retrieve precise information from the database",
        "description": "The SQL agent is specialized in interacting with databases and executing SQL queries."
    },
    "ResponseGenerator": {
        "name": "ResponseGenerator",
        "capabilities": "Generate human-readable responses based on RAG output",
        "description": "Формирует понятные ответы на основе полученных данных."
    }
}

# Основной блок для запуска скрипта
if __name__ == "__main__":
    logger.info("Запуск скрипта")

    while True:
        user_query = input("Введите ваш вопрос (или 'выход' для завершения): ")
        if user_query.lower() == 'выход':
            logger.info("Завершение работы скрипта")
            break

        logger.info(f"Получен запрос пользователя: {user_query}")
        result = supervisor.execute_task(user_query)

        if "error" in result:
            print(f"Произошла ошибка: {result['error']}")
            logger.error(f"Ошибка при обработке запроса: {result['error']}")
        else:
            print(f"\nОтвет: {result['result']}")
            print(f"\nДомен: {result['domain']}")
            print(f"Причина выбора домена: {result['reasoning']}")
            logger.info(f"Ответ пользователю: {result['result']}")
            logger.info(f"Использованный домен: {result['domain']}")
            logger.info(f"Причина выбора домена: {result['reasoning']}")

        print("\n" + "-" * 50 + "\n")

logger.info("Скрипт завершил работу")