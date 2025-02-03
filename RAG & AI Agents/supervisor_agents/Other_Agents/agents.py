import requests
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

departments = [
    {'category': "Претензия"},
    {'category': "Жалоба"},
    {'category': "Заявление"},
    {'category': "Судебное решение"},
    {'category': "Общая категория"},
]


class Agent:
    def __init__(self, name: str, capabilities: str):
        self.name = name
        self.capabilities = capabilities

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")


class SupportAgent(Agent):
    def __init__(self, name: str, capabilities: str, departments: List[Dict[str, str]], llm_url: str, api_key: str,
                 api_url: str, username: str, password: str):
        super().__init__(name, capabilities)
        self.departments = departments
        self.llm_url = llm_url
        self.api_key = api_key
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

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            logger.error("RAGAgent: Ошибка: не удалось авторизоваться")
            return None
        else:
            query = task['query']

            categories = "\n".join([department['category'] for department in self.departments])
            system_message = f"""As a customer support agent, you receive a message from a customer. 
            Please analyze the content of the message to determine the appropriate category and topic for their inquiry. The possible categories are:
            {categories}
            Based on the identified category, please consider the following departments that handle those inquiries:
            Отдел досудебного урегулирования (решает вопросы по категориям "Претензия" и "Жалоба")
            Отдел судебного урегулирования (решает вопросы по категориям "Заявление" и "Судебное решение")
            After categorizing the inquiry, remember to respond in Russian."""

            user_message = f"Запрос: {query}"

            schema = {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "department": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["category", "department", "reasoning"]
            }

            request_data = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "model": "llama-3-8b-instruct-8k",
                "max_tokens": 1000,
                "temperature": 0.1,
                "guided_json": json.dumps(schema),
                "guided_decoding_backend": "lm-format-enforcer"
            }

            logger.info(f"MailAgent: Отправка запроса к LLM: {json.dumps(request_data, ensure_ascii=False)}")

            response = requests.post(self.llm_url, json=request_data, headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'}
            )

            logger.info(f"SupportAgent: Получен ответ от LLM. Статус: {response.status_code}")
            logger.info(f"SupportAgent: Текст ответа: {response.text}")

            if response.status_code == 200:
                llm_response = json.loads(response.json()['choices'][0]['message']['content'])
                logger.info(f"SupportAgent: Обработанный ответ LLM: {json.dumps(llm_response, ensure_ascii=False)}")

                selected_department = next(
                    (department for department in self.departments if
                    department['category'] == llm_response['category']), None)

                if selected_department:
                    result = {
                        "category": selected_department["category"],
                        "department": llm_response["department"],
                        "reasoning": llm_response["reasoning"]
                    }
                else:
                    default_department = next(
                        department for department in self.departments if department["category"] == "Общая категория")
                    result = {
                        "category": default_department["category"],
                        "department": default_department["department"],
                        "reasoning": default_department["reasoning"]
                    }

                logger.info(f"SupportAgent: Результат обработки: {json.dumps(result, ensure_ascii=False)}")
                return result
            else:
                logger.error(f"SupportAgent: Ошибка при отправке запроса к LLM. Код ответа: {response.status_code}")
                logger.error(f"SupportAgent: Текст ответа: {response.text}")

                default_department = next(
                    department for department in self.departments if department["category"] == "Общая категория")
                result = {
                    "category": default_department["category"],
                    "department": "",
                    "reasoning": ""
                }
                logger.info(f"SupportAgent: Использована категория по умолчанию: {json.dumps(result, ensure_ascii=False)}")
                return result


class MailAgent(Agent):
    def __init__(self, name: str, capabilities: str, llm_url: str, api_key: str,
                 api_url: str, username: str, password: str):
        super().__init__(name, capabilities)
        self.departments = departments
        self.llm_url = llm_url
        self.api_key = api_key
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

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            logger.error("RAGAgent: Ошибка: не удалось авторизоваться")
            return None
        else:
            query = task['query']
            category = task['category']
            department = task['department']

            system_message = f"""Write the answer of response mail request of category {category} and name of department: {department}"""

            user_message = f"Запрос: {query}"

            schema = {
                "type": "object",
                "properties": {
                    "mail_theme": {"type": "string"},
                    "mail_text": {"type": "string"},
                },
                "required": ["mail_theme", "mail_text"]
            }

            request_data = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "model": "llama-3-8b-instruct-8k",
                "max_tokens": 10000,
                "temperature": 0.0,
                "guided_json": json.dumps(schema),
                "guided_decoding_backend": "lm-format-enforcer"
            }

            logger.info(f"MailAgent: Отправка запроса к LLM: {json.dumps(request_data, ensure_ascii=False)}")

            response = requests.post(self.llm_url, json=request_data, headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'}
            )

            logger.info(f"MailAgent: Получен ответ от LLM. Статус: {response.status_code}")
            logger.info(f"MailAgent: Текст ответа: {response.text}")

            if response.status_code == 200:
                llm_response = json.loads(response.json()['choices'][0]['message']['content'])
                logger.info(f"MailAgent: Обработанный ответ LLM: {json.dumps(llm_response, ensure_ascii=False)}")

                if llm_response:
                    result = {
                        "mail_theme": llm_response["mail_theme"],
                        "mail_text": llm_response["mail_text"]
                    }
                else:
                    '''
                    default_department = next(
                        department for department in self.departments if department["category"] == "Общая категория")
                    '''
                    result = {
                        "mail_theme": "",
                        "mail_text": ""
                    }

                logger.info(f"MailAgent: Результат обработки: {json.dumps(result, ensure_ascii=False)}")
                return result
            else:
                logger.error(f"MailAgent: Ошибка при отправке запроса к LLM. Код ответа: {response.status_code}")
                logger.error(f"MailAgent: Текст ответа: {response.text}")
                '''
                default_department = next(
                    department for department in self.departments if department["category"] == "Общая категория")
                '''
                result = {
                    "mail_theme": "",
                    "mail_text": ""
                }
                logger.info(f"MailAgent: Возвращено письмо по умолчанию: {json.dumps(result, ensure_ascii=False)}")
                return result

class ResponseGenerator(Agent):
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        support_response = task['rag_response']
        logger.info(f"ResponseGenerator: Получен ответ от RAG: {json.dumps(support_response, ensure_ascii=False)}")
        if support_response:
            result = {"result": support_response}
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
            support_result = self.agents['SupportAgent'].process({"query": task})
            logger.info(f"Supervisor: Результат маршрутизации: {json.dumps(support_result, ensure_ascii=False)}")

            # Шаг 2: RAG запрос

            mail_result = self.agents['MailAgent'].process({
                "category": support_result['category'],
                "department": support_result['department'],
                "query": task
            })
            logger.info(f"Supervisor: Результат RAG запроса: {json.dumps(mail_result, ensure_ascii=False)}")


            # Шаг 3: Генерация ответа
            response = self.agents['ResponseGenerator'].process({
                "router_response": support_result,
                "rag_response": mail_result
            })
            logger.info(f"Supervisor: Сгенерированный ответ: {json.dumps(response, ensure_ascii=False)}")

            result = {
                "category": support_result['category'],
                "department": support_result['department'],
                "reasoning": support_result['reasoning'],
                "query": task,
                "result": response['result']
            }
            logger.info(f"Supervisor: Итоговый результат: {json.dumps(result, ensure_ascii=False)}")
            return result
        except Exception as e:
            logger.error(f"Supervisor: Ошибка при выполнении задачи: {str(e)}", exc_info=True)
            return {"error": str(e)}


# Инициализация агентов и супервизора
support_agent = SupportAgent("SupportAgent", "Receive a message from a customer and return the name of department and response's category", departments,
                           config['llm']['url'], config['llm']['api_key'], config['api']['url'], config['api']['username'], config['api']['password'])
mail_agent = MailAgent("MailAgent", "Write an answer according to information from customer's message",
                     config['llm']['url'], config['llm']['api_key'], config['api']['url'], config['api']['username'], config['api']['password'])
response_generator = ResponseGenerator("ResponseGenerator", "Generate human-readable responses based on RAG output")

supervisor = Supervisor([support_agent, mail_agent, response_generator])

AGENT_DESCRIPTIONS = {
    "SupportAgent": {
        "name": "SupportAgent",
        "capabilities": "Receive a message from a customer and return the name of department and response's category",
        "description": "Получает сообщение от пользователя и возвращает название подразделения и категорию обращения."
    },
    "MailAgent": {
        "name": "MailAgent",
        "capabilities": "Write an answer according to information from customer's message",
        "description": "Пишет ответ, исходя из информации из письма покупателя."
    },
    "ResponseGenerator": {
        "name": "ResponseGenerator",
        "capabilities": "Generate human-readable responses based on SupportAgent and MailAgent outputs",
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
            print(f"\nКатегория: {result['category']}")
            print(f"Причина выбора категории: {result['reasoning']}")
            logger.info(f"Ответ пользователю: {result['result']}")
            logger.info(f"Использованная категория: {result['category']}")
            logger.info(f"Причина выбора категории: {result['reasoning']}")

        print("\n" + "-" * 50 + "\n")

logger.info("Скрипт завершил работу")

{'mail_theme': 'Претензия в порядке досудебного урегулирования спора',
 'mail_text': 'Отдел досудебного урегулирования Уважаемый/ая [Ф.И.О. представителя ООО «СЗ «А101»],'
              'В соответствии с Федеральным законом от 30.12.2004 г. № 214-ФЗ «Об участии в долевом строительстве многоквартирных домов и иных объектов недвижимости» и в связи с выявленными недостатками объекта долевого строительства, переданным Участнику в соответствии с Договором участия в долевом строительстве № ДИ99К-1.1-13, мы направляем настоящее требование о соразмерном уменьшении цены Договора.'
              'В соответствии с экспертным заключением № 24-1007/1 от 08.10.2024 г., величина затрат на восстановительный ремонт указанной квартиры составляет 873 449,89 руб. Согласно части 2 статьи 7 Федерального закона № 214-ФЗ от 30.12.2004 г., в случае если объект долевого строительства построен с отступлениями от условий договора и (или) указанных в части 1 настоящей статьи обязательных требований, приведшими к ухудшению качества такого объекта, или с иными недостатками, которые делают его непригодным для предусмотренного договором использования, участник долевого строительства вправе потребовать от застройщика соразмерного уменьшения цены договора.'
              'На основании изложенного настоящим требуем:'
              'Соразмерно уменьшить цену Договора на сумму в размере 873 449,89 руб. и выплатить данную сумму Участнику посредством ее перечисления по указанным ниже реквизитам.'
              'Реквизиты для перечисления денежных средств:'
              '[введите реквизиты]'
              'В случае если в течение 10-ти дней со дня получения настоящего требования застройщик не исполнит требования Участника, Участник вправе обратиться в суд с иском о взыскании указанной суммы.'
              'С уважением,'
              '[В.В. Романова]'
              'Участник Договора участия в долевом строительстве № ДИ99К-1.1-13'
              '[дата]'}
