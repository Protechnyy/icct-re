import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_PATH = "/home/users/lhy/OneKE/models/Qwen3-32B"


def main():
    print("start load", flush=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    print("loaded", flush=True)

    messages = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "请用一句中文说明你已成功加载。"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print("response:", response, flush=True)


if __name__ == "__main__":
    main()
