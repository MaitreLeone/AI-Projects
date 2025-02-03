import json
import requests
from typing import List, Dict, Any
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AGENT_DESCRIPTIONS = {
    "RAG": {
        "name": "RAG",
        "capabilities": "Information retrieval and analysis from the knowledge base",
        "description": "The RAG agent specializes in searching and analyzing information within a structured knowledge base."
    },
    "SQL": {
        "name": "SQL",
        "capabilities": "Execute SQL queries to retrieve precise information from the database",
        "description": "The SQL agent is specialized in interacting with databases and executing SQL queries."
    },
    "ResponseGenerator": {
        "name": "ResponseGenerator",
        "capabilities": "Generate human-readable responses based on raw data",
        "description": "The ResponseGenerator agent transforms raw data into natural language responses."
    }
}


class Agent:
    def __init__(self, name: str, capabilities: str):
        self.name = name
        self.capabilities = capabilities

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses")


class SQLAgent(Agent):
    def __init__(self, name: str, capabilities: str):
        super().__init__(name, capabilities)
        self.db = {
            "sales": [
                {"id": 1, "company_name": "Супервизор", "sale_date": "2023-05-15", "amount": 100000, "status": "completed"},
                {"id": 2, "company_name": "Супервизор", "sale_date": "2023-08-22", "amount": 150000, "status": "completed"},
                {"id": 3, "company_name": "Другая Компания", "sale_date": "2023-07-01", "amount": 200000, "status": "completed"}
            ],
            "purchases": [
                {"id": 1, "company_name": "Супервизор", "purchase_date": "2023-04-10", "amount": 80000, "status": "received"},
                {"id": 2, "company_name": "Супервизор", "purchase_date": "2023-07-05", "amount": 120000, "status": "received"},
                {"id": 3, "company_name": "Другой Поставщик", "purchase_date": "2023-06-15", "amount": 90000, "status": "received"}
            ]
        }
        self.last_executed_query = None

    def get_table_names(self) -> List[str]:
        return list(self.db.keys())

    def get_table_structure(self, table_name: str) -> Dict[str, List[str]]:
        if table_name in self.db:
            return {table_name: list(self.db[table_name][0].keys())}
        #добавлена ветка else
        else:
            return {}

    def execute_sql_query(self, query: str) -> Dict[str, Any]:
        self.last_executed_query = query
        logging.info(f"Executing SQL query: {query}")
        normalized_query = query.upper()

        for table_name in self.db.keys():
            if f"FROM {table_name.upper()}" in normalized_query:
                #имя компании вынести в отдельную переменную
                if "WHERE COMPANY_NAME = 'СУПЕРВИЗОР'" in normalized_query:
                    date_field = 'sale_date' if table_name == 'sales' else 'purchase_date'
                    if "MAX" in normalized_query or "LAST" in normalized_query:
                        #date_field = 'sale_date' if table_name == 'sales' else 'purchase_date'
                        result = max(item[date_field] for item in self.db[table_name] if item['company_name'] == 'Супервизор')
                    else:
                        #date_field = 'sale_date' if table_name == 'sales' else 'purchase_date'
                        result = [item[date_field] for item in self.db[table_name] if item['company_name'] == 'Супервизор']
                else:
                    result = self.db[table_name]

                logging.info(f"SQL query result: {result}")
                return {"result": result}

        logging.warning(f"Query not recognized: {query}")
        return {"result": None, "error": f"Query not recognized: {query}"}

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        query = task['query']
        table_names = self.get_table_names()
        selected_table = task.get('selected_table', table_names[0])  # Default to first table if not specified
        table_structure = self.get_table_structure(selected_table)

        # Generate SQL query (in a real scenario, this would be more sophisticated)
        sql_query = f"SELECT * FROM {selected_table} WHERE company_name = 'Супервизор'"

        return self.execute_sql_query(sql_query)


class RAGAgent(Agent):
    def __init__(self, name: str, capabilities: str):
        super().__init__(name, capabilities)
        self.knowledge_base = {
            "shipping_rules": """
            Правила отгрузки товаров компании "Супервизор":
            1. Минимальный заказ для отгрузки - 100 000 рублей.
            2. Отгрузка производится только по предоплате.
            3. Сроки отгрузки: 3-5 рабочих дней после поступления оплаты.
            4. Доставка осуществляется транспортными компаниями за счет покупателя.
            5. При заказе от 500 000 рублей доставка до терминала транспортной компании - бесплатно.
            6. Возврат товара принимается в течение 14 дней при сохранении товарного вида.
            7. Гарантийный срок на продукцию - 12 месяцев с момента отгрузки.
            """
        }

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"RAG agent is processing the task: {task}")
        query = task['query']
        #Возможно вынести как переменную экземпляра класса (перед __init__)
        shipping_keywords = ["правила отгрузки", "shipping rules", "отгрузка", "доставка", "shipping", "delivery"]

        if any(keyword in query.lower() for keyword in shipping_keywords):
            logging.info(f"Keyword match found for query: {query}")
            return {"result": self.knowledge_base["shipping_rules"]}
        else:
            logging.info(f"No keyword match found for query: {query}")
            return {
                "result": "Запрошенная информация не найдена. Вот все доступные данные в базе знаний:",
                "available_data": self.knowledge_base
            }


class ResponseGenerator(Agent):
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"ResponseGenerator is processing the task: {task}")
        data = task['data']
        original_task = task.get('original_task', '')

        if data.get('result'):
            if isinstance(data['result'], str):
                response = f"В ответ на ваш запрос '{original_task}': \n\n{data['result']}"
            elif isinstance(data['result'], list):
                #непонятно, зачем были получены даты (упоминания в коде нет)
                dates = ", ".join(data['result'])
                response = f"В ответ на ваш вопрос '{original_task}': Компания 'Супервизор' имеет следующие даты: {dates}."
            #подумать, какие данные могут храниться в result
            else:
                response = f"В ответ на ваш вопрос '{original_task}': Последняя дата для компании 'Супервизор' - {data['result']}."
        elif data.get('available_data'):
            response = f"На ваш запрос '{original_task}' не найдено точного соответствия, но вот доступная информация:\n\n"
            for key, value in data['available_data'].items():
                response += f"{key}:\n{value}\n\n"
        else:
            response = f"К сожалению, на ваш вопрос '{original_task}' не найдено информации. Результат запроса: {data.get('error', 'Неизвестная ошибка')}"

        logging.info(f"Generated response: {response}")
        return {"result": response}


class Supervisor:
    def __init__(self, url: str, agents: List[Agent]):
        self.url = url
        self.agents = {agent.name: agent for agent in agents}
        self.last_request_time = 0
        self.min_request_interval = 1

    def make_llm_request(self, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> Dict[str, Any]:
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)

        request_data = {
            "messages": messages,
            "model": "llama-3.1-8b-instruct",
            "max_tokens": 2000,
            "temperature": 0.1,
            "guided_json": json.dumps(schema),
            "guided_decoding_backend": "lm-format-enforcer"
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, json=request_data, timeout=30)
                response.raise_for_status()
                self.last_request_time = time.time()
                llm_response = json.loads(response.json()['choices'][0]['message']['content'])
                print(f"LLM Response: {llm_response}")
                return llm_response
            except requests.RequestException as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error("All attempts to request LLM failed")
                    raise
                time.sleep(2 ** attempt)

    def analyze_task(self, task: str) -> Dict[str, Any]:
        logging.info(f"Supervisor is analyzing the task: {task}")
        agents_info = "\n".join([f"{name}: {agent.capabilities}" for name, agent in self.agents.items()])

        schema = {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": list(self.agents.keys())},
                "task": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "additional_info": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            "required": ["agent", "task"]
        }

        messages = [
            {"role": "system", "content": f"""You are a supervisor who analyzes tasks and decides which agent to assign them to. Here are the agents' capabilities:
{agents_info}
Choose the most suitable agent and formulate the task for them. 
Remember:
- RAG agent is best for general information retrieval, especially for rules, policies, and non-structured data.
- SQL agent is for specific data queries about sales, purchases, dates, and structured data.
- Always prioritize using the RAG agent for general information and rules, and the SQL agent only for specific data queries."""},
            {"role": "user", "content": task}
        ]

        result = self.make_llm_request(messages, schema)
        print(f"Task Analysis Result: {result}")
        return result

    def execute_task(self, task: str) -> Dict[str, Any]:
        try:
            analysis = self.analyze_task(task)
            agent_name = analysis['agent']
            agent_task = analysis['task']

            print(f"Selected agent: {agent_name}")
            print(f"Agent task: {agent_task}")

            logging.info(f"Supervisor selected agent {agent_name} to execute the task")

            if agent_name not in self.agents:
                logging.error(f"Unknown agent: {agent_name}")
                return {"error": f"Unknown agent: {agent_name}"}

            result = self.agents[agent_name].process(agent_task)

            logging.info(f"Received result from agent {agent_name}: {result}")

            response_generator_result = self.agents["ResponseGenerator"].process({
                "data": result,
                "original_task": task
            })
            logging.info(f"Generated response: {response_generator_result}")
            result = response_generator_result

            return {
                "agent": agent_name,
                "initial_task": task,
                "agent_task": agent_task,
                "result": result
            }
        #попытаться вывести тип ошибки
        except Exception as e:
            logging.error(f"Error executing task: {e}")
            return {"error": str(e)}


# Initialize agents and supervisor
rag_agent = RAGAgent(AGENT_DESCRIPTIONS["RAG"]["name"], AGENT_DESCRIPTIONS["RAG"]["capabilities"])
sql_agent = SQLAgent(AGENT_DESCRIPTIONS["SQL"]["name"], AGENT_DESCRIPTIONS["SQL"]["capabilities"])
response_generator = ResponseGenerator(AGENT_DESCRIPTIONS["ResponseGenerator"]["name"],
                                       AGENT_DESCRIPTIONS["ResponseGenerator"]["capabilities"])

supervisor = Supervisor(
    "http://62.68.146.223/v1/chat/completions",
    [rag_agent, sql_agent, response_generator]
)