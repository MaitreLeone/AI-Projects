import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(base_model_path: str, adapter_path: str,  output_dir: str):
    """
    Функция слияния адаптера и модели
    
    :param base_model_path: путь к исходной модели
    :param adapter_path: путь к обученному адаптеру
    :param output_dir: путь к выходным файлам слияния
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.config.use_cache = True

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    base_model_path = "model"
    adapter_path = "adapter/out"
    output_dir = "trained"
    merge(base_model_path, adapter_path, output_dir)


if __name__ == "__main__":
    main()
