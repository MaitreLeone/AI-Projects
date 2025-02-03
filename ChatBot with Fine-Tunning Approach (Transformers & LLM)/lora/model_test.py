from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from util.io import read_jsonl
import json
import random
from tqdm import tqdm

def model_tokenizer_load(base_model, adapter_model=None, tokenizer_path=None):
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map="auto"
    )
    if adapter_model:        
        model = PeftModel.from_pretrained(model, adapter_model)        
        tokenizer = LlamaTokenizer.from_pretrained(adapter_model)
    else:
        if tokenizer_path:
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model.eval()
    
    return model, tokenizer
        

model, tokenizer = model_tokenizer_load(
                                    base_model = "models/LLaMa/HF/7B",
                                    adapter_model="llama_rualpaca_vokrugsveta_1888_3_epoch_trainer_640"
)

original_records = read_jsonl("vs_test.jsonl")
with open("ru_alpaca_qaknw.json") as r:
    templates = json.load(r)   
records = list()
for record in tqdm(original_records):
    question = record["question"]
    knowledge = record["knowledge"]
    out = record["answer"]    
    prompt_template = random.choice(templates["prompts_input"])
    input = prompt_template.format(question=question.strip(), knowledge=knowledge.strip())
    target = out.strip()
    records.append({"input":input, "target":target})

results = list()
for x in tqdm(records):
    input = tokenizer(x["input"], return_tensors="pt").to(model.device)
    output_ids = model.generate(
                        **input,
                        max_new_tokens=100)[0]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    #print(f"ANSWER:\t {answer}")
    #print(f"TARGET:\t {x['target']}")
    results.append({"input":x['input'], "output": output, "answer": output.split('Ответ:')[1].strip(), "target": x["target"]})
    
with open(f'trained_vs_model.json', 'w', encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)