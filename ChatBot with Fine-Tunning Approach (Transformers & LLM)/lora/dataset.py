import random
import json
from typing import Optional
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm


class QAKnwDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        templates_path: str,
        sample_rate: float = 1.0,
        only_target_loss: bool = True,
        use_padding: bool = False
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.only_target_loss = only_target_loss
        self.use_padding = use_padding
        self.is_printed = False

        with open(templates_path) as r:
            self.templates = json.load(r)

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_record(record)
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_record(self, record):
        question = record["question"]
        knowledge = record["knowledge"]
        out = record["answer"]
        templates = self.templates["prompts_input"]
        prompt_template = random.choice(templates)
        source = prompt_template.format(question=question.strip(), knowledge=knowledge.strip())
        target = out.strip()
        if not self.is_printed:
            print("Source and target examples")
            print(source)
            print(target)
            self.is_printed = True
        return self.convert_causal(source, target)

    def convert_causal(self, source, target=None):
        source_tokens = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True
        )["input_ids"]
        if self.tokenizer.bos_token_id:
            source_tokens.insert(0, self.tokenizer.bos_token_id)
        input_ids = source_tokens[:]
        actual_length = len(input_ids)
        max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2
        if target is not None:
            target_tokens = self.tokenizer(
                target,
                add_special_tokens=False,
                max_length=self.max_target_tokens_count,
                padding=False,
                truncation=True
            )["input_ids"]
            input_ids += target_tokens + [self.tokenizer.eos_token_id]
            actual_length = len(input_ids)
            if self.use_padding:
                padding = [self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)]
                input_ids.extend(padding)

        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())
        if self.use_padding:
            labels[actual_length:] = -100
            attention_mask[actual_length:] = 0
        if self.only_target_loss:
            labels[:len(source_tokens)] = -100
        assert input_ids.size(0) == labels.size(0) == attention_mask.size(0) <= max_length

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }