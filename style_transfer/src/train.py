import json
from pathlib import Path
from typing import Dict

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from dvclive.huggingface import DVCLiveCallback
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)


PROMPT_TEMPLATES = {
    "ru": "### Instruction:\nПерепиши исходный текст в стиле автора \"{author}\". Обрами результаты тегами <text> в начале и </text> в конце.\n\nИсходный текст:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>",
    "en": "### Instruction:\nRewrite the original text in the style of the author \"{author}\". Frame the results with <text> tags at the beginning and </text> at the end.\n\nOriginal text:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>",
    "uk": "### Instruction:\nПерепишіть оригінальний текст у стилі автора \"{author}\". Обрамте результати тегами <text> на початку та </text> у кінці.\n\nОригінальний текст:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>",
    "fr": "### Instruction:\nRéécrire le texte original dans le style de l'auteur \"{author}\". Encadrez les résultats avec les balises <text> au début et </text> à la fin.\n\nTexte original:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>",
    "de": "### Instruction:\nSchreiben Sie den Originaltext im Stil des Autors \"{author}\" um. Rahmen Sie die Ergebnisse mit <text>-Tags am Anfang und </text> am Ende ein.\n\nOriginaltext:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>",
    "es": "### Instruction:\nReescribe el texto original al estilo del autor \"{author}\". Enmarque los resultados con etiquetas <text> al principio y </text> al final.\n\nTexto original:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>",
    "tr": "### Instruction:\nOrijinal metni yazarın \"{author}\" tarzında yeniden yazın. Sonuçları başında <text> ve sonunda </text> etiketleriyle çerçeveleyin.\n\nOrjinal metin:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>",
    "ar": "### Instruction:\nأعد كتابة النص الأصلي بأسلوب المؤلف \"{author}\". ضع إطارًا للنتائج باستخدام علامات <text> في البداية و</text> في النهاية.\n\nالنص الأصلي:\n{neutral}\n\n### Response:\n<text>{stylized}</text></s>"
}

BATCH_SIZE = 2
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING = True


def get_dataset(dataset_path: str) -> DatasetDict:
    """
    Функция формирования датасета переноса стиля
    
    :param dataset_path: путь к папке с авторскими пересказами. Каждый файл содержит ключи "author", "lang" и "examples".
    
    :returns: объединенный датасет авторских пересказов с полями "lang", "author", "text", "paraphrase"
    """
    author_paths = Path(dataset_path).rglob("*.json")
    datasets = list()
    for author_path in author_paths:
        with open(author_path, mode="r", encoding="utf-8") as file:
            data = json.load(file)
        author = data["author"]
        lang = data["lang"]
        dataset = Dataset.from_list(data["examples"]).train_test_split(test_size=0.1)
        dataset["train"] = dataset["train"].add_column("author", [author]*len(dataset["train"]))
        dataset["train"] = dataset["train"].add_column("lang", [lang]*len(dataset["train"]))
        dataset["test"] = dataset["test"].add_column("author", [author]*len(dataset["test"]))
        dataset["test"] = dataset["test"].add_column("lang", [lang]*len(dataset["test"]))
        datasets.append(dataset)
    
    common_dataset = DatasetDict({
        "train": concatenate_datasets([dataset["train"] for dataset in datasets]),
        "test": concatenate_datasets([dataset["test"] for dataset in datasets])
    })
    common_dataset = common_dataset.shuffle()
    return common_dataset


def format_prompts(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Функция формирования промптов переноса стиля для языковой модели 
    
    :param example: элемент датасета переноса стиля
    :param tokenizer: токенизатор модели переноса стиля
    
    :returns: токенизированный промпт для элемента датасета с полями "input_ids" и "attention_mask"
    """
    output_texts = []
    for i in range(len(example["text"])):
        text = PROMPT_TEMPLATES[example["lang"][i]].format(author=example["author"][i], neutral=example["paraphrase"][i], stylized=example["text"][i])
        output_texts.append(text)
    return tokenizer(output_texts)


def train(base_model_path: str, dataset_path: str, output_dir: str, params_path: str):
    """
    Функция обучения LoRA-модуля переноса стиля для языковой модели
    
    :param base_model_path: путь к исходной модели и токенизатору
    :param dataset_path: путь к авторским пересказам
    :param output_dir: путь к выходным файлам обучения
    :param params_path: путь к файлу с параметрами обучения
    """
    with open(params_path, mode="r", encoding="utf-8") as file:
        params = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    dataset = get_dataset(dataset_path)
    dataset = dataset.map(lambda data: format_prompts(data, tokenizer), 
                          remove_columns=dataset["train"].column_names, 
                          batched=True, batch_size=1000)
    
    lora_config = LoraConfig(
        r = 64,
        lora_alpha = 16,
        lora_dropout = 0.1,
        bias = "none",
        task_type = TaskType.CAUSAL_LM
    )

    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=params["num_epochs"],
        learning_rate=params["learning_rate"],
        logging_strategy="epoch",
        fp16=True,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        callbacks=[DVCLiveCallback()]
    )
    
    if EARLY_STOPPING:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    
    trainer.train() 
    
    model.save_pretrained(Path(output_dir) / "out")
    tokenizer.save_pretrained(Path(output_dir) / "out")


def main():

    base_model_path = "model"
    dataset_path = "data"
    output_dir = "adapter"
    params = "config/params.json"
    train(base_model_path, dataset_path, output_dir, params)


if __name__ == "__main__":
    main()
