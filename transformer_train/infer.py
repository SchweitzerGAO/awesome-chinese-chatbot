from transformers import AutoModel
from transformers import AutoTokenizer
import torch
from peft import PeftModel
import infer_config

net = AutoModel.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
net = PeftModel.from_pretrained(net, './saved_models/epoch_3')


def inference(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.LongTensor([input_ids])
    out = net.generate(
        input_ids=input_ids,
        max_length=infer_config.max_length,
        temperature=infer_config.temperature,
        top_p=infer_config.top_p,
    )
    answer = tokenizer.decode(out[0])\
                      .replace(text, '')\
                      .replace('\nEND', '').strip()
    return answer


def chat():
    history = []
    with torch.no_grad():
        while True:
            text = input('User: ')
            text = 'Instruction: ' + text + 'Answer: '
            history.append(text)
            text_with_history = '\n'.join(history[-3:])
            answer = inference(text_with_history)
            history.append(f'Answer: {answer}')


if __name__ == '__main__':
    chat()
