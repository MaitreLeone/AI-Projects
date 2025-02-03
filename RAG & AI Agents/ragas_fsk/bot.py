import asyncio
import os
import signal
import subprocess
from datetime import datetime

import yaml
from telebot.async_telebot import AsyncTeleBot
from telebot.types import BotCommand
import aiofiles  # Импортируем aiofiles для асинхронного чтения файлов

# Функция для загрузки конфигурации из файла config.yaml
def load_configuration():
    with open("Новые результаты/config.yaml", "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

# Загрузка конфигурации
config = load_configuration()

api_key = config["api_key"]
base_url = config["base_url"]
telegram_token = config["telegram_token"]

# Инициализация бота
bot = AsyncTeleBot(telegram_token)

# Список для хранения запущенных процессов
processes = []

# Флаг для отслеживания запущенного теста
test_running = False

# Асинхронная функция для выполнения скриптов и отправки результатов
async def run_scripts_and_send_results(chat_id):
    global test_running
    scripts = [
        "1_test_rag.py",
        "2_evaluated.py",
        "3_scatter_plot.py",
        "4_recomendation.py"
    ]

    message = await bot.send_message(chat_id, "Запуск всех скриптов...")
    print("Запуск всех скриптов...")  # Отладочное сообщение

    for script in scripts:
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=message.message_id, text=f"Запуск {script}...")
            print(f"Запуск {script}...")  # Отладочное сообщение
            log_file = f"{script}.log"
            with open(log_file, "w") as f:
                process = await asyncio.create_subprocess_exec(
                    "python", script,
                    stdout=f,
                    stderr=f
                )
                processes.append(process)

            # Чтение файла в реальном времени
            await asyncio.create_task(stream_log_to_telegram(log_file, chat_id, message.message_id, script, process))

            await process.wait()
            print(f"{script} завершен с кодом {process.returncode}")  # Отладочное сообщение
            if process.returncode == 0:
                await bot.edit_message_text(chat_id=chat_id, message_id=message.message_id, text=f"{script} завершен.")
            else:
                await bot.edit_message_text(chat_id=chat_id, message_id=message.message_id, text=f"Произошла ошибка при выполнении {script}.")
                test_running = False
                return
        except Exception as e:
            await bot.edit_message_text(chat_id=chat_id, message_id=message.message_id, text=f"Произошла ошибка при выполнении {script}:\n{str(e)}")
            print(f"Произошла ошибка при выполнении {script}: {str(e)}")  # Отладочное сообщение
            test_running = False
            return

    # Отправка результатов
    await send_results(chat_id)
    test_running = False

# Асинхронная функция для чтения файла и обновления сообщения в Telegram
async def stream_log_to_telegram(log_file, chat_id, message_id, script, process):
    print(f"Чтение логов для {script}...")  # Отладочное сообщение
    async with aiofiles.open(log_file, "r") as f:
        last_message = None
        while True:
            line = await f.readline()
            if not line:
                if process.returncode is not None:  # Проверка завершения процесса
                    break
                await asyncio.sleep(1)
                continue
            new_message = f"{script} выполняется...\n{line.strip()}"
            if new_message != last_message:
                await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=new_message)
                last_message = new_message
            print(new_message)  # Отладочное сообщение

# Функция для проверки, пуст ли файл
def is_file_empty(file_path):
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

# Асинхронная функция для отправки результатов
async def send_results(chat_id):
    print("Отправка результатов...")  # Отладочное сообщение

    # Отправка последнего файла из папки 5_recomendation_result
    result_directory = '5_recomendation_result'
    if os.path.exists(result_directory) and os.listdir(result_directory):
        latest_file = find_latest_file(result_directory)
        if not is_file_empty(latest_file):
            with open(latest_file, 'rb') as file:
                await bot.send_document(chat_id, file)
        else:
            print(f"Файл {latest_file} пустой. Пропуск отправки рекомендаций.")  # Отладочное сообщение
    else:
        print(f"Директория {result_directory} не существует или пуста.")  # Отладочное сообщение

    # Отправка scatter диаграммы из папки 3_scatter_result
    scatter_directory = '3_scatter_result'
    if os.path.exists(scatter_directory) and os.listdir(scatter_directory):
        latest_scatter_file = find_latest_file(scatter_directory)
        with open(latest_scatter_file, 'rb') as file:
            await bot.send_document(chat_id, file)
    else:
        print(f"Директория {scatter_directory} не существует или пуста.")  # Отладочное сообщение

# Функция для поиска самого последнего файла в папке
def find_latest_file(directory):
    print(f"Поиск последнего файла в {directory}...")  # Отладочное сообщение
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    latest_file = max(files, key=os.path.getctime)
    print(f"Найден последний файл: {latest_file}")  # Отладочное сообщение
    return latest_file

# Обработчик команды /start_test
@bot.message_handler(commands=['start_test'])
async def handle_start_test(message):
    global test_running
    chat_id = message.chat.id
    print(f"Получена команда /start_test от пользователя {chat_id}")  # Отладочное сообщение
    if test_running:
        await bot.send_message(chat_id, "Тест уже запущен. Пожалуйста, дождитесь его завершения.")
        return
    test_running = True
    await bot.send_message(chat_id, "Пожалуйста, пришлите файл qa_dataset_fsk_reindexed.json.")

# Обработчик команды /stop
@bot.message_handler(commands=['stop'])
async def handle_stop(message):
    global test_running
    chat_id = message.chat.id
    print(f"Получена команда /stop от пользователя {chat_id}")  # Отладочное сообщение
    stop_all_processes()
    await bot.send_message(chat_id, "Все запущенные процессы остановлены.")
    print("Команда /stop получена")  # Отладочное сообщение
    test_running = False

# Обработчик команды /check_config
@bot.message_handler(commands=['check_config'])
async def handle_check_config(message):
    chat_id = message.chat.id
    print(f"Получена команда /check_config от пользователя {chat_id}")  # Отладочное сообщение
    with open("Новые результаты/config.yaml", 'rb') as file:
        await bot.send_document(chat_id, file)
    print("Команда /check_config получена")  # Отладочное сообщение

# Обработчик команды /load_config
@bot.message_handler(commands=['load_config'])
async def handle_load_config(message):
    chat_id = message.chat.id
    print(f"Получена команда /load_config от пользователя {chat_id}")  # Отладочное сообщение
    await bot.send_message(chat_id, "Пожалуйста, пришлите файл config.yaml.")

@bot.message_handler(content_types=['document'])
async def handle_document(message):
    global test_running
    chat_id = message.chat.id
    document_name = message.document.file_name
    print(f"Получен файл от пользователя {chat_id}: {document_name}")  # Отладочное сообщение
    if document_name == "config.yaml":
        try:
            file_info = await bot.get_file(message.document.file_id)
            downloaded_file = await bot.download_file(file_info.file_path)
            with open("Новые результаты/config.yaml", 'wb') as new_file:
                new_file.write(downloaded_file)
            print("Файл config.yaml обновлен")  # Отладочное сообщение
            await bot.send_message(chat_id, "Файл config.yaml успешно обновлен.")
            # Перезагрузка конфигурации
            global config, api_key, base_url, telegram_token
            config = load_configuration()
            api_key = config["api_key"]
            base_url = config["base_url"]
            telegram_token = config["telegram_token"]
            print("Конфигурация перезагружена")  # Отладочное сообщение
        except Exception as e:
            print(f"Ошибка при обновлении config.yaml: {str(e)}")  # Отладочное сообщение
            await bot.send_message(chat_id, f"Произошла ошибка при обновлении config.yaml: {str(e)}")
    elif document_name == "qa_dataset_fsk_reindexed.json":
        if test_running:
            try:
                file_info = await bot.get_file(message.document.file_id)
                downloaded_file = await bot.download_file(file_info.file_path)
                dataset_path = 'dataset/qa_dataset_fsk_reindexed.json'
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                with open(dataset_path, 'wb') as new_file:
                    new_file.write(downloaded_file)
                print(f"Файл сохранен по пути: {dataset_path}")  # Отладочное сообщение
                await bot.send_message(chat_id, "Файл получен и сохранен. Запуск теста...")
                await run_scripts_and_send_results(chat_id)
            except Exception as e:
                print(f"Ошибка при обработке файла: {str(e)}")  # Отладочное сообщение
                await bot.send_message(chat_id, f"Произошла ошибка при обработке файла: {str(e)}")
                test_running = False
        else:
            await bot.send_message(chat_id, "Пожалуйста, сначала запустите тест с помощью команды /start_test.")
    else:
        await bot.send_message(chat_id, "Пожалуйста, пришлите файл в формате документа.")

# Функция для остановки всех запущенных процессов
def stop_all_processes():
    print("Остановка всех процессов...")  # Отладочное сообщение
    for process in processes:
        try:
            process.terminate()
            print(f"Процесс {process.pid} остановлен")  # Отладочное сообщение
        except OSError as e:
            print(f"Ошибка при остановке процесса {process.pid}: {str(e)}")  # Отладочное сообщение
    processes.clear()

# Обработчик сигнала для остановки бота и всех процессов
def handle_exit(signum, frame):
    print("Получен сигнал для завершения работы")  # Отладочное сообщение
    stop_all_processes()
    exit(0)

# Асинхронная функция для установки команд бота и запуска polling
async def main(argv):
    await bot.set_my_commands([
        BotCommand("start_test", "Запустить тест"),
        BotCommand("stop", "Остановить все процессы"),
        BotCommand("check_config", "Проверить конфигурацию"),
        BotCommand("load_config", "Загрузить новый config.yaml"),
        BotCommand("help", "Помощь"),
    ])
    await bot.polling(skip_pending=True, non_stop=True, request_timeout=60)

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    print("Бот запущен")  # Отладочное сообщение
    asyncio.run(main(None))