import requests
from requests.auth import HTTPBasicAuth
import json
import base64
import yaml
from typing import Dict, Any, List
import logging
import sys

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
    {"domain_name": "ЖК(жилищный комплекс) 1(первый)  Донской", "domain_id": "6704d450dae88f75d8ae9985",
     "chat_id": "6704d451dae88f75d8ae9986"},
    {"domain_name": "ЖК(жилищный комплекс) 1(первый) Измайловский", "domain_id": "6704d45cdae88f75d8ae9987",
     "chat_id": "6704d45edae88f75d8ae9988"},
    {"domain_name": "ЖК(жилищный комплекс) 1(первый) Ленинградский", "domain_id": "6704d46bdae88f75d8ae9989",
     "chat_id": "6704d46cdae88f75d8ae998a"},
    {"domain_name": "ЖК(жилищный комплекс) 1(первый) Южный", "domain_id": "6704d477dae88f75d8ae998b",
     "chat_id": "6704d478dae88f75d8ae998c"},
    {"domain_name": "ЖК(жилищный комплекс) Amber city", "domain_id": "6704d489dae88f75d8ae998d",
     "chat_id": "6704d48adae88f75d8ae998e"},
    {"domain_name": "ЖК(жилищный комплекс) Южная битца", "domain_id": "6704d495dae88f75d8ae998f",
     "chat_id": "6704d497dae88f75d8ae9990"},
    {"domain_name": "Общая база знаний", "domain_id": "6704d4ebdae88f75d8ae9991", "chat_id": "6704d4eddae88f75d8ae9992"}
]


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

        domains_list = ", ".join([f"{d['domain_name']} (chat_id: {d['chat_id']})" for d in self.domains_info])
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
                    "domain_name": selected_domain['domain_name'],
                    "domain_id": selected_domain['domain_id'],
                    "chat_id": selected_domain['chat_id'],
                    "reasoning": llm_response['reasoning']
                }
            else:
                default_domain = next(
                    domain for domain in self.domains_info if domain['domain_name'] == "Общая база знаний")
                result = {
                    "domain_name": default_domain['domain_name'],
                    "domain_id": default_domain['domain_id'],
                    "chat_id": default_domain['chat_id'],
                    "reasoning": "Выбран домен по умолчанию, так как указанный chat_id не найден."
                }

            logger.info(f"RouterAgent: Результат обработки: {json.dumps(result, ensure_ascii=False)}")
            return result
        else:
            logger.error(f"RouterAgent: Ошибка при отправке запроса к LLM. Код ответа: {response.status_code}")
            logger.error(f"RouterAgent: Текст ответа: {response.text}")

            default_domain = next(
                domain for domain in self.domains_info if domain['domain_name'] == "Общая база знаний")
            result = {
                "domain_name": default_domain['domain_name'],
                "domain_id": default_domain['domain_id'],
                "chat_id": default_domain['chat_id'],
                "reasoning": "Выбран домен по умолчанию из-за ошибки в процессе выбора."
            }
            logger.info(f"RouterAgent: Использован домен по умолчанию: {json.dumps(result, ensure_ascii=False)}")
            return result


class RAGAgent(Agent):
    def __init__(self, name: str, capabilities: str, api_url: str, username: str, password: str):
        super().__init__(name, capabilities)
        self.api_url = api_url
        self.username = username
        self.password = password
        self.token = self.login()

    def login(self):
        credentials = base64.b64encode(f"{self.username}:{self.password}".encode('utf-8')).decode('utf-8')
        logger.info(f"RAGAgent: Попытка авторизации для пользователя {self.username}")
        response = requests.post(f"{self.api_url}/sign_in",
                                 headers={'accept': 'application/json', 'Authorization': f'Basic {credentials}'})

        if response.status_code == 200 or response.status_code == 201:
            logger.info("RAGAgent: Авторизация успешна")
            return response.json().get("access_token")
        else:
            logger.error(f"RAGAgent: Ошибка авторизации. Код ответа: {response.status_code}")
            logger.error(f"RAGAgent: Текст ответа: {response.text}")
            return None

    def send_query(self, chat_id: str, question: str):
        if not self.token:
            logger.error("RAGAgent: Ошибка: не удалось авторизоваться")
            return None

        headers = {'Authorization': f'Bearer {self.token}'}
        url = f'{self.api_url}/search/{chat_id}'
        payload = {
            "text": question,
            "return_debug_data": True,
            "meta_override": meta_override_config,
        }

        logger.info(f"RAGAgent: Отправка запроса. URL: {url}")
        logger.info(f"RAGAgent: Payload: {json.dumps(payload, ensure_ascii=False)}")

        response = requests.post(url, headers=headers, json=payload)

        logger.info(f"RAGAgent: Получен ответ. Статус: {response.status_code}")
        logger.info(f"RAGAgent: Текст ответа: {response.text}")

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"RAGAgent: Ошибка при отправке запроса. Код ответа: {response.status_code}")
            logger.error(f"RAGAgent: Текст ответа: {response.text}")
            return None

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        chat_id = task['chat_id']
        query = task['query']
        logger.info(f"RAGAgent: Обработка запроса. Chat ID: {chat_id}, Query: {query}")
        result = self.send_query(chat_id, query)
        logger.info(f"RAGAgent: Результат обработки: {json.dumps(result, ensure_ascii=False)}")
        return result


class ResponseGenerator(Agent):
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        rag_response = task['rag_response']
        logger.info(f"ResponseGenerator: Получен ответ от RAG: {json.dumps(rag_response, ensure_ascii=False)}")
        if rag_response and 'response' in rag_response:
            result = {"result": rag_response['response']}
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

            # Шаг 2: RAG запрос
            rag_result = self.agents['RAGAgent'].process({
                "chat_id": router_result['chat_id'],
                "query": task
            })
            logger.info(f"Supervisor: Результат RAG запроса: {json.dumps(rag_result, ensure_ascii=False)}")

            # Шаг 3: Генерация ответа
            response = self.agents['ResponseGenerator'].process({
                "rag_response": rag_result
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
rag_agent = RAGAgent("RAGAgent", "Retrieve information from knowledge base",
                     config['api']['url'], config['api']['username'], config['api']['password'])
response_generator = ResponseGenerator("ResponseGenerator", "Generate human-readable responses based on RAG output")

supervisor = Supervisor([router_agent, rag_agent, response_generator])

AGENT_DESCRIPTIONS = {
    "RouterAgent": {
        "name": "RouterAgent",
        "capabilities": "Route queries to appropriate knowledge domains",
        "description": "Выбирает подходящий домен знаний на основе запроса пользователя."
    },
    "RAGAgent": {
        "name": "RAGAgent",
        "capabilities": "Retrieve information from knowledge base",
        "description": "Ищет информацию в выбранном домене знаний."
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