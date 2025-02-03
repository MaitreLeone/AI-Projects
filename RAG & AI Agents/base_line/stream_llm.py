import streamlit as st
import time
from supervisor_system import supervisor, AGENT_DESCRIPTIONS

st.set_page_config(layout="wide")
st.title("Интерактивный супервизор задач")

# Краткое описание системы
st.write("""
Эта система демонстрирует работу супервизора и агентов на базе модели llama-3.1 8b. 
Она анализирует ваш запрос, выбирает подходящего агента, выполняет задачу и генерирует ответ.
""")

# Функция для создания значков агентов
def create_agent_badges():
    cols = st.columns(len(AGENT_DESCRIPTIONS))
    badges = {}
    for i, (agent, info) in enumerate(AGENT_DESCRIPTIONS.items()):
        with cols[i]:
            badges[agent] = st.empty()
    return badges

# Функция для обновления значков агентов
def update_agent_badges(badges, active_agent=None):
    for agent, badge in badges.items():
        color = "green" if agent == active_agent else "gray"
        badge.markdown(f"<h3 style='text-align: center; color: {color};'>{agent}</h3>", unsafe_allow_html=True)

# Создаем значки агентов
agent_badges = create_agent_badges()

# Initialize session state for task if it doesn't exist
if 'task' not in st.session_state:
    st.session_state.task = ""

# User input
task = st.text_input("Введите задачу:", value=st.session_state.task)

# Add example tasks as suggestions
st.write("Примеры задач:")
example_tasks = [
    "Какие правила отгрузки товаров у компании 'Супервизор'?",
    "Когда была последняя продажа проекта компании 'Супервизор'?",
    "Перечислите все даты продаж проектов 'Супервизор'.",
    "Когда была последняя закупка товаров компанией 'Супервизор'?"
]
for example_task in example_tasks:
    if st.button(example_task, key=example_task):
        st.session_state.task = example_task
        st.rerun()

if st.button("Выполнить задачу"):
    if task:
        # Placeholder for results
        task_analysis = st.empty()
        agent_selection = st.empty()
        task_execution = st.empty()
        query_generation = st.empty()
        query_execution = st.empty()
        response_generation = st.empty()
        final_result = st.empty()

        # Execute the task step by step
        try:
            # Шаг 1: Анализ задачи
            task_analysis.subheader("Анализ задачи")
            task_analysis.write("Анализируем задачу...")
            update_agent_badges(agent_badges, "Supervisor")
            analysis = supervisor.analyze_task(task)
            task_analysis.json(analysis)
            time.sleep(1)

            # Шаг 2: Выбор агента
            agent_selection.subheader("Выбранный агент")
            selected_agent = analysis['agent']
            update_agent_badges(agent_badges, selected_agent)
            agent_selection.write(f"Имя: {selected_agent}")
            agent_selection.write(f"Возможности: {AGENT_DESCRIPTIONS[selected_agent]['capabilities']}")
            agent_selection.write(f"Описание: {AGENT_DESCRIPTIONS[selected_agent]['description']}")
            time.sleep(1)

            # Шаг 3: Выполнение задачи
            task_execution.subheader("Выполнение задачи")
            task_execution.write("Выполняем задачу...")
            result = supervisor.agents[selected_agent].process(analysis['task'])
            task_execution.json(result)
            time.sleep(1)

            # Дополнительный шаг для SQL агента
            if selected_agent == "SQL":
                query_generation.subheader("Генерация SQL-запроса")
                query_generation.write("Генерируем SQL-запрос...")
                # Получаем SQL-запрос из результата выполнения задачи
                sql_query = getattr(supervisor.agents[selected_agent], 'last_executed_query', None)
                if sql_query:
                    query_generation.code(sql_query, language="sql")
                else:
                    query_generation.write("SQL-запрос не был сгенерирован или недоступен")
                time.sleep(1)

                query_execution.subheader("Выполнение SQL-запроса")
                query_execution.write("Выполняем SQL-запрос...")
                query_result = result.get('result', 'Результат SQL-запроса отсутствует')
                query_execution.json(query_result)
                time.sleep(1)

            # Шаг 4: Генерация ответа
            response_generation.subheader("Генерация ответа")
            response_generation.write("Генерируем окончательный ответ...")
            update_agent_badges(agent_badges, "ResponseGenerator")
            response = supervisor.agents["ResponseGenerator"].process({
                "data": result,
                "original_task": task
            })
            response_generation.json(response)
            time.sleep(1)

            # Шаг 5: Финальный результат
            final_result.subheader("Финальный результат")
            final_result.write(response['result'])

            # Сброс значков агентов
            update_agent_badges(agent_badges)

        except Exception as e:
            st.error(f"Произошла ошибка при выполнении задачи: {str(e)}")

# Sidebar content
st.sidebar.header("Схема агентов")
agent_diagram = """
digraph G {
    bgcolor="transparent";
    node [shape=rect, style="filled,rounded", fontname="Arial", fontcolor="white", color="#2a2a2a", fontsize=14];
    edge [color="#4a4a4a", penwidth=1.5];

    Supervisor [label="Supervisor", fillcolor="#FF69B4"];
    RAG [label="RAG Agent", fillcolor="#4682B4"];
    SQL [label="SQL Agent", fillcolor="#32CD32"];
    RG [label="Response Generator", fillcolor="#FFA07A"];
    KB [label="Knowledge Base", shape=cylinder, fillcolor="#8A2BE2"];
    DB [label="Database", shape=cylinder, fillcolor="#20B2AA"];

    Supervisor -> RAG [label=""];
    Supervisor -> SQL [label=""];
    Supervisor -> RG [label=""];
    RAG -> KB [label="query/retrieve"];
    SQL -> DB [label="query/retrieve"];
    RAG -> RG [label="pass data"];
    SQL -> RG [label="pass data"];
}
"""
st.sidebar.graphviz_chart(agent_diagram)

st.sidebar.header("Информация об агентах")
for agent, info in AGENT_DESCRIPTIONS.items():
    st.sidebar.subheader(agent)
    st.sidebar.text(f"Возможности: {info['capabilities']}")
    st.sidebar.text(f"Описание: {info['description']}")

st.sidebar.header("О системе")
st.sidebar.write("""
Эта система использует различных агентов для обработки ваших запросов:
- RAG агент ищет информацию в базе знаний
- SQL агент выполняет запросы к базе данных
- ResponseGenerator формирует понятные ответы на основе полученных данных

Система анализирует вашу задачу, выбирает подходящего агента, выполняет задачу и генерирует ответ.
""")