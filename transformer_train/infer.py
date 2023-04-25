from transformers import AutoModel
from transformers import AutoTokenizer
import torch
from peft import LoraConfig, TaskType, get_peft_model
import infer_config

torch.set_default_tensor_type(torch.cuda.HalfTensor)
net = AutoModel.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
net = get_peft_model(net, peft_config)
net.load_state_dict(torch.load('./saved_models/epoch_3/adapter_model.bin'), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def inference(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.LongTensor([input_ids])
    input_ids = input_ids.to(infer_config.device)
    out = net.generate(
        input_ids=input_ids,
        max_length=infer_config.max_length,
        temperature=infer_config.temperature,
        top_p=infer_config.top_p,
        repetition_penalty=infer_config.repetition_penalty
    )
    answer = tokenizer.decode(out[0])
    return answer


def chat():
    with torch.no_grad():
        while True:
            text = input('User: ')
            answer = inference(text)
            print(f'XiaoRuan: {answer}')


if __name__ == '__main__':
    chat()
