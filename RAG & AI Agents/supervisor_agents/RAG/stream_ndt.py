import streamlit as st
import time
from supervisor_ndt import supervisor, AGENT_DESCRIPTIONS, domains_info

st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .sidebar .element-container {
        word-wrap: break-word;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("Интерактивный супервизор задач")

# Краткое описание системы
st.write("""
Эта система демонстрирует работу супервизора и агентов на базе модели llama-3.1 8b. 
Она анализирует ваш запрос, выбирает подходящий домен знаний, выполняет задачу и генерирует ответ.
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

# User input with search icon
task = st.text_input("Введите задачу:", value=st.session_state.task, placeholder="🔍 Поиск...")

# Example tasks as suggestions
st.write("Примеры задач:")
example_tasks = [
    "Сколько машиномест в паркинге ЖК 1 Измайловский?",
    "Размер окон в ЖК 1 Измайловский?",
    "Как добраться личным транспортом до 1 Измайловский?",
    "Как добраться общественным транспортом до 1-го Измайловского?",
    "Есть ли подземный паркинг в 1 Измайловском?",
    "Как добраться на общественном транспорте до ЖК 1й Донской?",
    "Сроки сдачи Амбер Сити?",
    "Количество парковочных мест в Амбер?",
    "Когда выдают ключи в Амбер?",
    "Адрес ЖК Амбер Сити?",
    "Какой класс строительства в ЖК 1 Южный?",
    "Окружение ЖК 1 Южный?",
    "Сколько км от МКАД находится Амбер Сити?",
    "Расположение Амбер от ТТК?"
]
cols = st.columns(3)  # Разделим на 3 колонки для более компактного отображения
for i, example_task in enumerate(example_tasks):
    if cols[i % 3].button(example_task, key=example_task):
        st.session_state.task = example_task
        st.rerun()

# Зеленая кнопка "Выполнить задачу"
if st.button("Выполнить задачу", type="primary"):
    if task:
        # Placeholder for results
        domain_selection = st.empty()
        router_analysis = st.empty()
        rag_execution = st.empty()
        response_generation = st.empty()
        final_result = st.empty()

        # Execute the task step by step
        try:
            # Шаг 1: Маршрутизация
            update_agent_badges(agent_badges, "RouterAgent")
            router_result = supervisor.agents['RouterAgent'].process({"query": task})
            domain_selection.subheader(f"Выбранный домен знаний: {router_result['domain_name']}")

            with router_analysis.expander("Детали анализа и выбора домена", expanded=False):
                st.json(router_result)
            time.sleep(1)

            # Шаг 2: RAG запрос
            update_agent_badges(agent_badges, "RAGAgent")
            rag_result = supervisor.agents['RAGAgent'].process({
                "chat_id": router_result['chat_id'],
                "query": task
            })
            with rag_execution.expander("Детали выполнения RAG запроса", expanded=False):
                st.json(rag_result)
            time.sleep(1)

            # Шаг 3: Генерация ответа
            update_agent_badges(agent_badges, "ResponseGenerator")
            response = supervisor.agents["ResponseGenerator"].process({
                "rag_response": rag_result
            })
            with response_generation.expander("Детали генерации ответа", expanded=False):
                st.json(response)
            time.sleep(1)

            # Шаг 4: Финальный результат
            final_result.subheader("Финальный результат")
            final_result.write(response['result'])

            # Сброс значков агентов
            update_agent_badges(agent_badges)

        except Exception as e:
            st.error(f"Произошла ошибка при выполнении задачи: {str(e)}")

# Left Sidebar content
with st.sidebar:
    with st.expander("Схема агентов", expanded=False):
        agent_diagram = """
        digraph G {
            bgcolor="transparent";
            node [shape=rect, style="filled,rounded", fontname="Arial", fontcolor="white", color="#2a2a2a", fontsize=14];
            edge [color="#4a4a4a", penwidth=1.5];

            Supervisor [label="Supervisor", fillcolor="#FF69B4"];
            Router [label="Router Agent", fillcolor="#FFA500"];
            RAG [label="RAG Agent", fillcolor="#4682B4"];
            RG [label="Response Generator", fillcolor="#FFA07A"];
            KB [label="Knowledge Base", shape=cylinder, fillcolor="#8A2BE2"];

            Supervisor -> Router [label="1. Запрос"];
            Router -> Supervisor [label="2. Выбор домена"];
            Supervisor -> RAG [label="3. Запрос к домену"];
            RAG -> KB [label="4. Поиск информации"];
            KB -> RAG [label="5. Результаты поиска"];
            RAG -> Supervisor [label="6. Результаты RAG"];
            Supervisor -> RG [label="7. Генерация ответа"];
            RG -> Supervisor [label="8. Финальный ответ"];
        }
        """
        st.graphviz_chart(agent_diagram)

    with st.expander("Информация об агентах", expanded=False):
        for agent, info in AGENT_DESCRIPTIONS.items():
            st.subheader(agent)
            st.write(f"**Возможности:** {info['capabilities']}")
            st.write(f"**Описание:** {info['description']}")
            st.write("---")

    with st.expander("Доступные домены знаний", expanded=False):
        for domain in domains_info:
            st.write(f"• {domain['domain_name']}")

    with st.expander("О системе", expanded=False):
        st.write("""
        Эта система использует следующих агентов для обработки ваших запросов:

        - RouterAgent выбирает подходящий домен знаний
        - RAGAgent ищет информацию в выбранном домене знаний
        - ResponseGenerator формирует понятные ответы на основе полученных данных

        Система анализирует вашу задачу, выбирает подходящий домен, выполняет поиск и генерирует ответ.
        """)