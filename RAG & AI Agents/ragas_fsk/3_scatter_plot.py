import json
import pandas as pd
import plotly.graph_objects as go
import textwrap
import os
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Функция для извлечения числового значения из строки и приведения его к процентам
def extract_number(text):
    try:
        return float(text) * 20  # Приведение к процентам (5-балльная шкала к 100%)
    except ValueError:
        return None


# Функция для переноса строки в тексте
def wrap_text(text, width=150):
    return '<br>'.join(textwrap.wrap(text, width=width))


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Функция для записи данных с оценками ниже 4 в файл
def write_low_scores_to_file(data, directory='./', filename=None):
    if filename is None:
        filename = f'./4_low_scores_result/low_scores_{current_time}.json'
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    low_scores = [item for item in data if any(score is not None and score < 40 for score in [
        extract_number(item['Evaluation']['Answer Relevancy']),
        extract_number(item['Evaluation']['Context Relevancy'])
    ])]

    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(low_scores, file, ensure_ascii=False, indent=4)


# Функция для построения графика
def build_graph(input_filename, output_directory='./'):
    with open(input_filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    evaluation_data = []

    for item in data:
        evaluation = item['Evaluation']
        evaluation_data.append({
            'Question Number': item['Question Number'],
            'Question': wrap_text(item['Question']),
            'Answer': wrap_text(item['Answer']),
            'Ground Truth': wrap_text(item['Ground Truth']),
            'Answer Relevancy': extract_number(evaluation['Answer Relevancy']),
            'Context Relevancy': extract_number(evaluation['Context Relevancy']),
        })

    # Записать данные с оценками ниже 4 в файл
    write_low_scores_to_file(data, directory=output_directory)

    df = pd.DataFrame(evaluation_data)
    df['Average Score'] = df[['Answer Relevancy', 'Context Relevancy']].mean(axis=1)
    df['Average Score (%)'] = df['Average Score']
    overall_average_score = df['Average Score (%)'].mean()
    df['Hover Info'] = df.apply(lambda
                                    row: f"<b>Question Number:</b> {row['Question Number']}<br><b>Question:</b> {row['Question']}<br><b>Answer:</b> {row['Answer']}<br><b>Ground Truth:</b> {row['Ground Truth']}",
                                axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Question Number'],
        y=df['Answer Relevancy'],
        mode='markers',
        name='Answer Relevancy',
        customdata=df['Hover Info'],
        hovertemplate='%{customdata}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df['Question Number'],
        y=df['Context Relevancy'],
        mode='markers',
        name='Context Relevancy',
        customdata=df['Hover Info'],
        hovertemplate='%{customdata}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Scores for Each Question (Overall Average Score: {overall_average_score:.2f}%)',
        xaxis_title='Question Number',
        yaxis_title='Score (%)',
        xaxis={
            'categoryorder': 'total descending',
            'showgrid': True,
            'gridcolor': 'gray'
        },
        yaxis={
            'showgrid': True,
            'gridcolor': 'gray'
        },
        width=3000,
        template='plotly_dark',
        legend=dict(
            font=dict(size=16),
            bgcolor='rgba(0,0,0,0)'
        )
    )

    os.makedirs(output_directory, exist_ok=True)
    fig.write_html(os.path.join(output_directory, f"./3_scatter_result/scores_scatter_{current_time}.html"))

    # Вывод общего среднего балла в консоль
    logging.info(f"Overall Average Score: {overall_average_score:.2f}%")


# Функция для поиска самого последнего файла в папке
def find_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    latest_file = max(files, key=os.path.getctime)
    return latest_file


if __name__ == "__main__":
    logging.info("Начало выполнения 3_scatter_plot.py")
    input_directory = './2_evaluated_result'
    input_filename = find_latest_file(input_directory)
    build_graph(input_filename)
    print("Script execution completed.")
    logging.info("Завершение выполнения 3_scatter_plot.py")