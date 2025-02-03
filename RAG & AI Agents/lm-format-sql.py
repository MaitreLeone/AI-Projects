import streamlit as st
import json
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection parameters
DB_PARAMS = {
    "dbname": "fsk",
    "user": "admin",
    "password": "Lol770905",
    "host": "62.68.146.188",
    "port": "5432"
}

# URL of your vLLM server
VLLM_URL = "http://62.68.146.188/v1/chat/completions"

# Preloaded residential complexes data
RESIDENTIAL_COMPLEXES = [
    {"id": 1, "name": "ЖК 1 'Донской'", "chat_id": "6704d451dae88f75d8ae9986"},
    {"id": 2, "name": "ЖК 1 'Измайловский'", "chat_id": "6704d45edae88f75d8ae9988"},
    {"id": 3, "name": "ЖК 1 'Ленинградский'", "chat_id": "6704d46cdae88f75d8ae998a"},
    {"id": 4, "name": "ЖК 1 'Южный'", "chat_id": "6704d478dae88f75d8ae998c"},
    {"id": 5, "name": "ЖК 'Amber city'", "chat_id": "6704d48adae88f75d8ae998e"},
    {"id": 6, "name": "ЖК 'Южная Битца'", "chat_id": "6704d497dae88f75d8ae9990"}
]

# Database schema information
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
DB_SCHEMA += "\n".join([f"- {complex['name']} (chat_id: {complex['chat_id']})" for complex in RESIDENTIAL_COMPLEXES])

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

# New schema for pre-check
pre_check_schema = {
    "type": "object",
    "properties": {
        "generate_sql": {"type": "boolean"},
        "explanation": {"type": "string"}
    },
    "required": ["generate_sql", "explanation"]
}


def pre_check_query(user_prompt):
    system_prompt = f"""You are an AI assistant that determines whether a user's query requires generating an SQL query based on the following database schema:

    {DB_SCHEMA}

    Analyze the user's input and determine if it requires an SQL query to be answered.
    Return true if an SQL query is needed, false otherwise.
    Provide a brief explanation for your decision.
    """

    request_data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": "llama-3.1-8b-instruct",
        "max_tokens": 200,
        "temperature": 0.0,
        "guided_json": json.dumps(pre_check_schema),
        "guided_decoding_backend": "lm-format-enforcer"
    }

    response = requests.post(VLLM_URL, json=request_data)

    if response.status_code == 200:
        result = response.json()
        return json.loads(result['choices'][0]['message']['content'])
    else:
        st.error(f"Error in pre-check: {response.status_code}")
        st.json(response.json())
        return None


def get_sql_query(user_prompt):
    system_prompt = f"""You are an AI assistant that translates natural language queries into SQL. 
    Use the following database schema information to generate accurate SQL queries:

    {DB_SCHEMA}

    Generate SQL queries based on user natural language input.
    In the query field, return none if the query does not match the database schema
    Use parameterized queries with %s placeholders for security.
    Use chat_id to identify specific apartment complexes

    """

    request_data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": "llama-3.1-8b-instruct",
        "max_tokens": 500,
        "temperature": 0.0,
        "guided_json": json.dumps(sql_schema),
        "guided_decoding_backend": "lm-format-enforcer"
    }

    response = requests.post(VLLM_URL, json=request_data)

    if response.status_code == 200:
        result = response.json()
        st.json(result)
        return result['choices'][0]['message']['content']
    else:
        st.error(f"Error: {response.status_code}")
        st.json(response.json())
        return None


def execute_sql_query(query, params):
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
                        return f"Жилой комплекс с chat_id '{params[0]}' не найден в базе данных."

                    if "parking" in query.lower():
                        cur.execute(
                            "SELECT COUNT(*) FROM parking JOIN residential_complexes ON parking.complex_id = residential_complexes.id WHERE residential_complexes.chat_id = %s",
                            [params[0]])
                        parking_count = cur.fetchone()['count']
                        if parking_count == 0:
                            return f"Информация о паркинге для жилого комплекса с chat_id '{params[0]}' отсутствует."

                    return "Запрос выполнен успешно, но не вернул результатов. Возможно, данные отсутствуют."
                return results
    except psycopg2.Error as e:
        return f"Произошла ошибка при выполнении запроса: {e}"


def main():
    st.title("Система запросов на естественном языке к SQL")
    st.write("Вы можете задавать вопросы о жилых комплексах, квартирах и парковках.")

    # Сайдбар с информацией о базе данных и возможных вопросах
    st.sidebar.header("Информация о базе данных")
    st.sidebar.subheader("Содержимое базы данных:")
    st.sidebar.markdown("""
    - Жилые комплексы (названия, идентификаторы)
    - Квартиры (размер, количество комнат, уровень комфорта, размер окон, средняя цена за кв.м)
    - Парковки (количество мест)
    """)

    st.sidebar.subheader("Примеры вопросов:")
    st.sidebar.markdown("""
    - Сколько квартир в ЖК 'Донской'?
    - Какая средняя цена за квадратный метр в ЖК 'Amber city' для квартир с более чем 2 комнатами?
    - Сколько парковочных мест в ЖК 'Южная Битца'?
    - Какие жилые комплексы имеют квартиры площадью более 100 кв.м?
    - Какой уровень комфорта у квартир в ЖК 'Ленинградский'?
    """)

    if st.button("Показать доступные жилые комплексы"):
        st.subheader("Доступные жилые комплексы:")
        for complex in RESIDENTIAL_COMPLEXES:
            st.write(f"Название: {complex['name']}, Chat ID: {complex['chat_id']}")

    user_prompt = st.text_input("Введите ваш запрос на естественном языке:")

    if st.button("Выполнить запрос"):
        if user_prompt:
            # Первый этап: предварительная проверка
            pre_check_result = pre_check_query(user_prompt)
            if pre_check_result is None:
                st.error("Ошибка при выполнении предварительной проверки.")
            else:
                st.subheader("Результат предварительной проверки:")
                st.write(f"Требуется SQL-запрос: {'Да' if pre_check_result['generate_sql'] else 'Нет'}")
                st.write(f"Объяснение: {pre_check_result['explanation']}")

                if not pre_check_result['generate_sql']:
                    st.warning(
                        "Запрос не требует генерации SQL. Попробуйте переформулировать вопрос или задать другой.")
                else:
                    # Второй этап: генерация и выполнение SQL-запроса
                    st.subheader("Генерация SQL-запроса:")
                    agent_output = get_sql_query(user_prompt)
                    st.json(agent_output)

                    try:
                        sql_data = json.loads(agent_output)

                        st.subheader("Сгенерированный SQL-запрос:")
                        st.code(sql_data['query'])
                        st.write("Параметры:", sql_data['params'])
                        st.write("Объяснение:", sql_data['explanation'])

                        result = execute_sql_query(sql_data['query'], sql_data['params'])

                        st.subheader("Результат запроса:")
                        if isinstance(result, str):
                            st.write(result)
                        else:
                            st.json(result)
                    except json.JSONDecodeError:
                        st.error("Не удалось разобрать вывод агента как JSON. Вывод может быть не в ожидаемом формате.")
                    except KeyError as e:
                        st.error(f"В выводе агента отсутствует ожидаемый ключ: {e}")
        else:
            st.warning("Пожалуйста, введите запрос перед выполнением.")


if __name__ == "__main__":
    main()