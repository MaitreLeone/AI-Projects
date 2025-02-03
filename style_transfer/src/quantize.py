import json

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AwqConfig


def quantize(model_path: str, dataset_path: str, output_dir: str):
    """
    Функция квантования модели
    
    :param model_path: путь к исходной модели
    :param dataset_path: путь к датасету квантования
    :param output_dir: путь к выходным файлам квантования
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto")

    with open(dataset_path, mode="r", encoding="utf-8") as file:
        dataset = json.load(file)
    dataset = [elem for elem in dataset if len(elem) > 0 and len(elem.split(" ")) > 20]

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMV"}
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    model.quantize(tokenizer, quant_config=quant_config, calib_data=dataset)

    model.model.config.quantization_config = quantization_config

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    base_model_path = "trained"
    dataset_path = "src/quantization_dataset.json"
    output_dir = "quantized"
    quantize(base_model_path, dataset_path, output_dir)


if __name__ == "__main__":
    main()
