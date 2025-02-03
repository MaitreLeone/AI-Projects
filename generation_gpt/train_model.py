from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import random
import torch

# ---------------- SETTINGS SECTION ----------------

model_name = './models/rugpt3large_based_on_gpt2'

save_path = './model'

seed = 0                  # Параметр для воспроизводимости результатов
random_seed = True        # Случайное начальное значение генератора случайных чисел
block_size = 1024         # Размер блока для обучения - до 2048
test_size = 0.2           # Доля данных для валидации

# Параметры обучения

num_epochs = 1            # Количество эпох
learning_rate = 0.001     # Скорость обучения
weight_decay = 0.01       # Коэффициент регуляризации
batch_size = 6            # Размер батча
num_warmup_steps = 10     # Количество шагов для разогрева
num_logging_steps = 100
num_eval_steps = 10       # Количество шагов через которое запускается цикл оценки

# --------------- LOAD MODELS ----------------

def init():
    global tokenizer, model, data_collator
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<pad>', eos_token='<pad>')

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --------------------------------------------------

def group_texts(examples, block_size):
    
    # Конкатенируем все токены
    concatenated_examples = torch.tensor([x for xs in examples for x in xs])
    
    # Размер всей входной последовательности в токенах
    total_length = len(concatenated_examples)
    
    # Здесь отбрасывается остаток, не попавший в целое количество блоков. Можно было бы оставить его, дополнив <pad>
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # Разделение на блоки
    result = [concatenated_examples[i : i + block_size] for i in range(0, total_length, block_size)]

    return result


def process_data(data, tokenizer, block_size):
    texts = [item.get('text', '') for item in data]
    
    tokenized_texts = tokenizer(texts, truncation=False)

    tokenized_texts_blocks = {'input_ids': group_texts(tokenized_texts['input_ids'], block_size),
                              'attention_mask': group_texts(tokenized_texts['attention_mask'], block_size)}
    
    return tokenized_texts_blocks


def train_model(messages):
    if not random_seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    train, valid = train_test_split(messages, test_size=test_size, random_state=42)

    train_dataset = Dataset.from_dict(process_data(train, tokenizer, block_size))
    valid_dataset = Dataset.from_dict(process_data(valid, tokenizer, block_size))
    
    training_args = TrainingArguments(
        output_dir=save_path, # The output directory
        overwrite_output_dir=True, # Overwrite the content of the output directory
        num_train_epochs=num_epochs, # Number of training epochs
        logging_strategy='steps',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        do_eval=True,
        per_device_train_batch_size=batch_size, # Batch size for training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        logging_steps=num_logging_steps,
        eval_steps=num_eval_steps, # Number of update steps between two evaluations.
        warmup_steps=num_warmup_steps,# Number of warmup steps for learning rate scheduler
        prediction_loss_only=False,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    trainer.train()
