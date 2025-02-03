import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from dvclive.huggingface import DVCLiveCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig,
    TrainingArguments
)
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


PROMPT_TEMPLATES = {
    "ru": "Ниже приведена инструкция, описывающая задачу. Напиши ответ, который надлежащим образом завершает запрос.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "en": "Below are instructions describing the task. Write a response that properly completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "uk": "Нижче наведено інструкцію, яка описує завдання. Напиши відповідь, яка належним чином завершує запит.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "fr": "Vous trouverez ci-dessous des instructions décrivant la tâche. Écrivez une réponse qui complète correctement la requête.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "de": "Nachfolgend finden Sie Anweisungen, die die Aufgabe beschreiben. Schreiben Sie eine Antwort, die die Anfrage ordnungsgemäß abschließt.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "es": "A continuación encontrará instrucciones que describen la tarea. Escriba una respuesta que complete correctamente la solicitud.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "tr": "Aşağıda görevi açıklayan talimatlar bulunmaktadır. İsteği doğru şekilde tamamlayan bir yanıt yazın.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "ar": "فيما يلي تعليمات تصف المهمة. اكتب استجابة تكمل الطلب بشكل صحيح.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
}

BATCH_SIZE = 4
SAVE_TOTAL_LIMIT = 2


def get_dataset(dataset_path: str, test_size: float, debug: bool = False) -> DatasetDict:
    """
    Функция формирования датасета
    
    :param dataset_path: путь к папке с файла для обучения. Каждый файл содержит ключи "lang" и "examples".
    
    :returns: объединенный датасет с полями "lang", "instruction", "output"
    """
    dataset_paths = Path(dataset_path).rglob("*.json")
    datasets = list()
    for dataset_path in dataset_paths:
        with open(dataset_path, mode="r", encoding="utf-8") as file:
            data = json.load(file)
        lang = data["lang"]
        dataset = Dataset.from_list(data["examples"]).train_test_split(test_size=test_size)
        dataset["train"] = dataset["train"].add_column("lang", [lang]*len(dataset["train"]))
        dataset["test"] = dataset["test"].add_column("lang", [lang]*len(dataset["test"]))
        datasets.append(dataset)

    common_dataset = DatasetDict({
        "train": concatenate_datasets([dataset["train"] for dataset in datasets]),
        "test": concatenate_datasets([dataset["test"] for dataset in datasets])
    })

    common_dataset = common_dataset.shuffle()

    if debug:
        logger.info(dataset)
        logger.info(f"TRAIN dataset SAMPLE:\n{dataset['train'][0]}")
        logger.info(f"TEST dataset SAMPLE:\n{dataset['test'][0]}")

    return common_dataset


def format_prompts(example: Dict) -> List:
    """
    Функция формирования промптов
    
    :param example: элемент датасета
    
    :returns: промпт для элемента датасета
    """
    output_texts = []
    for i in range(len(example["instruction"])):
        text = PROMPT_TEMPLATES[example["lang"][i]].format(instruction=example["instruction"][i], output=example["output"][i])
        output_texts.append(text)
    return output_texts


def init_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    assert tokenizer, "Error of loading tokenizer"
    return tokenizer


def prepare_peft(model):
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
        
    return model, peft_config


def load_bnb_model(model_path: str):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=bnb_config,
                                                 device_map="auto",
                                                 use_cache=False,
                                                 torch_dtype=torch.bfloat16
    )
        
    return model


def load_gptq_model(model_path: str):
    gptq_config = GPTQConfig(bits=4,
                             disable_exllama=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             quantization_config = gptq_config,
                                             use_cache=False
    )
    
    return model


def load_whole_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 use_cache=False
    )
    
    return model


def init_model(model_path: str, bnb_quantization: bool = False, gptq_quantization: bool = False, debug: bool = False):
    model = None
    peft_config = None
    if bnb_quantization:
        if debug:
            logger.info("BNB QUANTIZATION and PEFT")
        model = load_bnb_model(model_path)
        model, peft_config = prepare_peft(model)
    elif gptq_quantization:
        if debug:
            logger.info("GPTQ QUANTIZATION and PEFT")
        model = load_gptq_model(model_path)
        model, peft_config = prepare_peft(model)
    else:
        if debug:
            logger.info("WHOLE MODEL")
        model = load_whole_model(model_path)
        model, peft_config = prepare_peft(model)
    assert model, "Error of loading model"

    return model, peft_config


def train(base_model_path: str, dataset_path: str, output_dir: str, params_path: str):
    """
    Функция обучения LoRA-модуля переноса стиля для языковой модели
    
    :param base_model_path: путь к исходной модели и токенизатору
    :param dataset_path: путь к датасету с новостями
    :param output_dir: путь к выходным файлам обучения
    :param params_path: путь к файлу с параметрами обучения
    """
    with open(params_path, mode="r", encoding="utf-8") as file:
        params = json.load(file)

    dataset = get_dataset(dataset_path, params["train_test_split"])

    tokenizer = init_tokenizer(base_model_path)
    model, peft_config = init_model(base_model_path)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=params["num_epochs"],
        learning_rate=params["learning_rate"],
        optim="adamw_torch",
        logging_strategy="epoch",
        fp16=True,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=False,
        formatting_func=format_prompts,
        dataset_batch_size=BATCH_SIZE,
        dataset_num_proc=8,
        args=training_arguments,
        callbacks=[DVCLiveCallback()]
    )

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
