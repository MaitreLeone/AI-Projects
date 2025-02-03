import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
import numpy as np
from pynvml import *
import time

import model_424 as model
import data_tuple as data
import evaluation

num_workers = 16
training_args = TrainingArguments(
                output_dir="trained_models",
                per_device_train_batch_size=16,
                gradient_accumulation_steps=1,
                per_device_eval_batch_size = 16,
                tf32=True
            )
torch.backends.cuda.matmul.allow_tf32 = training_args.tf32
accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps, logging_dir='logs')
logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
        filename='logs/training.log',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
def print_gpu_utilization():
    nvmlInit()
    n = torch.cuda.device_count()
    log = ""
    for i in range(n):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        log += f"GPU-{i} {info.used//1024**2} MB.\t"
    logger.info(log, main_process_only=True)

def evaluate(accelerator, fid_model, dataset, tokenizer, collator):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=training_args.per_device_eval_batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collator
    )
    fid_model.eval()
    dataloader = accelerator.prepare(dataloader)
    total = 0
    exactmatch = []
    fid_model = fid_model.module if hasattr(fid_model, "module") else fid_model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch
            outputs = fid_model.generate(
                input_ids=context_ids.to(accelerator.device),
                attention_mask=context_mask.to(accelerator.device),
                max_length=50
            )
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
    exactmatch = np.mean(exactmatch)
    return exactmatch

def t5_fid_train(t5_model_name, is_freeze_encoder=True, train_file="nq/dev.json", test_file="nq/test.json"):
    logger.info(f"Run t5_fid_train for \n {t5_model_name}", main_process_only=True)
    logger.info("-"*10, main_process_only=True)
    logger.info(f"train batch size: {training_args.per_device_train_batch_size}", main_process_only=True)
    logger.info(f"eval batch size: {training_args.per_device_eval_batch_size}", main_process_only=True)
    logger.info(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}", main_process_only=True)
    logger.info(f"usage tf32: {training_args.tf32}", main_process_only=True)
    logger.info("-"*10, main_process_only=True)

    torch.cuda.empty_cache()
    print_gpu_utilization()
    t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    fid_model = model.FiDT5(t5.config)
    fid_model.load_t5(t5.state_dict())

    if is_freeze_encoder:
        logger.info("Freeze encoder", main_process_only=True)
        for param in fid_model.encoder.parameters():
            param.requires_grad = False
    else:
        logger.info("NO freeze encoder", main_process_only=True)
        
    tokenizer = T5Tokenizer.from_pretrained(t5_model_name, use_fast=False, model_max_length=512)
    opt_text_maxlength = 200
    opt_answer_maxlength = 100
    collator = data.Collator(opt_text_maxlength, tokenizer, answer_maxlength=opt_answer_maxlength)
    opt_train_data = train_file
    train_examples = data.load_data(opt_train_data)
    opt_n_context = 100
    train_ds = data.Dataset(train_examples, opt_n_context)
    opt_eval_data = test_file
    eval_examples = data.load_data(opt_eval_data)
    eval_ds = data.Dataset(eval_examples, opt_n_context)
    
    torch.manual_seed(53)
    train_sampler = RandomSampler(train_ds)
    train_dataloader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=training_args.per_device_train_batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collator
    )
    
    opt_lr = 1e-4
    opt_weight_decay = 0.1
    optimizer = torch.optim.AdamW(fid_model.parameters(), lr=opt_lr, weight_decay=opt_weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step:1.0)

    fid_model, optimizer, train_dataloader, scheduler = accelerator.prepare(fid_model, optimizer, train_dataloader, scheduler)
    print_gpu_utilization()

    eval_freq_step = 400
    num_epochs = 10
    num_steps = -1
    model_root_path = f"fid_{t5_model_name.replace('/', '_')}_{opt_train_data.replace('/', '_').replace('.json', '')}"
    fid_model.train()
    for epoch in range(num_epochs):
        logger.info(f"Init epoch {epoch}", main_process_only=False)
        epoch_start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            if step == num_steps:
                break
            with accelerator.accumulate(fid_model):
                optimizer.zero_grad()
                (_, labels, _, p_ids, p_mask) = batch
                train_loss = fid_model(
                                        input_ids=p_ids,
                                        attention_mask=p_mask,
                                        labels=labels,
                                        return_dict=False
                                        )[0]
                accelerator.backward(train_loss)
                optimizer.step()
                scheduler.step()

                if step % eval_freq_step == 0:
                    dev_em = evaluate(accelerator, fid_model, eval_ds, tokenizer, collator)
                    log = f"{epoch} : {step} | "
                    log += f"train: {train_loss:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log, main_process_only=False)
                    fid_model.train()
        epoch_end_training_time = time.time()
        dev_em = evaluate(accelerator, fid_model, eval_ds, tokenizer, collator)
        epoch_end_evaluation_time = time.time()
        log = f"Final epoch {epoch} | train: {train_loss:.3f} | evaluation: {100*dev_em:.2f}EM | lr: {scheduler.get_last_lr()[0]:.5f}"
        log+= f" | epoch trainig time: {epoch_end_training_time - epoch_start_time}"
        log+= f" | epoch evaluation time: {epoch_end_evaluation_time - epoch_end_training_time}"
        logger.info(log, main_process_only=False)
        
        accelerator.wait_for_everyone()
        print_gpu_utilization()
        logger.info('Saving model', main_process_only=True)
        if accelerator.is_main_process:
            trained_fid_model = accelerator.unwrap_model(fid_model)
            trained_fid_model.save_pretrained(
                f"{training_args.output_dir}/{model_root_path}/epoch_{epoch}",
                save_function = accelerator.save
                )

def trained_fid_model_evaluate(model_path, model_name, device, eval_file="nq/test.json"):    
    logger.info(f"Run trained_fid_model_evaluate for \n {model_path}, \n {model_name} \n on {device}", main_process_only=True)
    logger.info("-"*10, main_process_only=True)
    logger.info(f"train batch size: {training_args.per_device_train_batch_size}", main_process_only=True)
    logger.info(f"eval batch size: {training_args.per_device_eval_batch_size}", main_process_only=True)
    logger.info(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}", main_process_only=True)
    logger.info(f"usage tf32: {training_args.tf32}", main_process_only=True)
    logger.info("-"*10, main_process_only=True)

    torch.cuda.empty_cache()
    print_gpu_utilization()   
    fid_model = model.FiDT5.from_pretrained(model_path).to(device)        
    tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False, model_max_length=512)
    opt_text_maxlength = 200
    opt_answer_maxlength = -1
    collator = data.Collator(opt_text_maxlength, tokenizer, answer_maxlength=opt_answer_maxlength)    
    opt_n_context = 100
    opt_eval_data = eval_file
    eval_examples = data.load_data(opt_eval_data)
    eval_ds = data.Dataset(eval_examples, opt_n_context)
    print_gpu_utilization()
    sampler = SequentialSampler(eval_ds)
    dataloader = DataLoader(eval_ds,
        sampler=sampler,
        batch_size=training_args.per_device_eval_batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collator
    )
    fid_model.eval()
    exactmatch = []
    fid_model = fid_model.module if hasattr(fid_model, "module") else fid_model
    time0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch
            outputs = fid_model.generate(
                input_ids=context_ids.to(device),
                attention_mask=context_mask.to(device),
                max_length=50
            )
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = eval_ds.get_example(idx[k])['answers']
                score = evaluation.ems(ans, gold)
                exactmatch.append(score)    
    exactmatch = np.mean(exactmatch)
    time1 = time.time()
    log = f"All steps: {i} | evaluation: {100*exactmatch:.2f}EM | time: {time1-time0}"
    logger.info(log)
    print_gpu_utilization()

if __name__ == "__main__": 
    try:
        trained_fid_model_evaluate(
            "/opt/miniconda3/.jupyter/lab/workspaces/ta_space/FiD_accelerate_training/trained_models/fid_sberbank-ai_ruT5-base_sberquad_fid_train/epoch_5",
            "sberbank-ai/ruT5-base",
            "cuda:1",
            "sberquad_fid_dev.json"
        )
    except Exception as err:        
        for err_arg in err.args:
            logger.info(f"ERROR: {err_arg}", main_process_only=False)
