from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def apply_lora(base_model_path: str, adapter_path: str, output_dir: str):
    """
    Функция слияния адаптера
    
    :param base_model_path: путь к исходной модели
    :param adapter_path: путь к обученному адаптеру
    :param output_dir: путь к выходным файлам
    """
    print(f"Loading the base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                      device_map="auto",
                                                      torch_dtype=torch.float16
                                                    )

    print(f"Loading the LoRA adapter from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, 
                                      adapter_path, 
                                      is_trainable=False
                                      )
    model.print_trainable_parameters()

    print("Applying the LoRA")
    model = model.merge_and_unload()        
    print(model)
    
    print(f"Saving the target model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    base_model_path = "model"
    adapter_path = "adapter/out"
    output_dir = "trained"
    apply_lora(base_model_path, adapter_path, output_dir)


if __name__ == "__main__":
    main()
