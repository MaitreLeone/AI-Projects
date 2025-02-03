from docx import Document
import json

def docx_to_json(docx_path, json_path):
    """
    Конвертирует DOCX файл в JSON файл.

    Args:
        docx_path: Путь к DOCX файлу.
        json_path: Путь для сохранения JSON файла.
    """

    try:
        document = Document(docx_path)
        paragraphs = []
        for paragraph in document.paragraphs:
            paragraphs.append(paragraph.text)

        json_data = {"paragraphs": paragraphs}
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        print(f"DOCX файл '{docx_path}' успешно конвертирован в JSON файл '{json_path}'.")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{docx_path}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    docx_file_path = "dataset/Вопросы для бота.docx"  # Замените на путь к вашему DOCX файлу
    json_file_path = "dataset/alfa_bank_dataset.json"  # Замените на желаемый путь к JSON файлу
    docx_to_json(docx_file_path, json_file_path)