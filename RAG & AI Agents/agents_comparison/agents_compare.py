import asyncio
import json
from typing import Tuple, Union, Dict, Optional, Any, List
import logging
import aiohttp

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

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


    def process_gigachat(self, task: Dict[str, Any]) -> Dict[str, Any]:
        query = task['query']

        domains_list = ", ".join([f'{d["name"]} (chat_id: {d["chat_id"]})' for d in self.domains_info])

        pass


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

        response = requests.post(self.api_url, json=request_data, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'})

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
        "model": "GigaChat",
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

async def main():
    api_url = ""
    query = ""
    async with aiohttp.ClientSession() as session:
        route_query(session, query)
        async with session.post(api_url) as response:
            pass


asyncio.run(main())