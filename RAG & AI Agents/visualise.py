from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

class Visualizer:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def visualize(self, method: str):
        raise Exception("Введён некорректный инструмент или метод для визуализации")

class PlotlyVisualizer(Visualizer):
    def find_jsonl_files(self):
        return list(self.directory.rglob('*.jsonl')) + list(self.directory.rglob('*.json'))

    def read_files(self):
        list_of_data = []
        filenames = self.find_jsonl_files()
        for filename in filenames:
            with open(filename, mode='r', encoding='utf-8') as file:
                if filename.suffix == '.jsonl':
                    data_vis = [json.loads(line) for line in file]
                else:  # .json
                    data_vis = [json.load(file)]
                list_of_data.append(data_vis)
        return list_of_data, filenames

    def create_data(self):
        list_of_data, filenames = self.read_files()
        data_vis_list = []
        for i, data in enumerate(list_of_data):
            if 'summary' in data[-1]:
                data_vis = {
                    'accuracy_mean': data[-1]['summary']['success'] / data[-1]['summary']['total'] * 100,
                    'time_mean': data[-1]['summary']['average_generation_time']
                }
            else:
                accuracy = [int(elem['success']) for elem in data]
                time = [int(elem['generation_time']) for elem in data]
                data_vis = {
                    'accuracy': accuracy,
                    'time': time,
                    'accuracy_mean': np.mean(accuracy) * 100,
                    'time_mean': np.mean(time),
                    'model_name': filenames[i].stem.split('_')
                }
            data_vis_list.append(data_vis)
        return data_vis_list, filenames

    def visualize(self, method: str):
        data, filenames = self.create_data()
        categories = ['time_mean', 'accuracy_mean']
        colors = ['green', 'red', 'yellow']
        subplot_info = [f"Результаты {file.name}" for file in filenames]

        if method == 'histogram':
            rows, cols = 2, 2
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_info)
            for i, elem in enumerate(data):
                values_vis = [elem[category] for category in categories if category in elem]
                row, col = divmod(i, cols)
                # Добавление сносок
                text_labels = [f"{value:.2f}" for value in values_vis]  # Форматирование значений
                fig.add_trace(
                    go.Bar(
                        x=categories,
                        y=values_vis,
                        marker_color=colors,
                        text=text_labels,  # Добавление текста
                        textposition='auto'  # Позиция текста
                    ),
                    row=row + 1, col=col + 1
                )
            fig.update_layout(
                title='Средние значения Time и Accuracy',
                showlegend=False
            )
            fig.write_image("average_values.png", height=1080, width=1920)
            fig.show()

def main():
    visualizer = PlotlyVisualizer('./results/')
    visualizer.visualize(method='histogram')

if __name__ == '__main__':
    main()