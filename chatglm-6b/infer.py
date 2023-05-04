from transformers import AutoModel
from transformers import AutoTokenizer
import torch
from peft import LoraConfig, TaskType, get_peft_model
import infer_config
import argparse

meta_instruction = '你的名字是小软，是基于开源语言模型在党史问答数据集上微调的党史问答机器人，你可以与用户闲聊，回答与党史相关的问题，但你不擅长做数学题和角色扮演' \
                   '注意在回答问题时，如果答案超出了最大字数，则记住剩余的答案，当用户说“继续”时继续输出剩余的答案。以下是用户的输入，你需要做出回答：\n'

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


def inference(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.LongTensor([input_ids])
    input_ids = input_ids.to(infer_config.device)
    out = net.generate(
        input_ids=input_ids,
        temperature=infer_config.temperature,
        top_p=infer_config.top_p,
        repetition_penalty=infer_config.repetition_penalty,
        max_new_tokens=infer_config.max_new_tokens
    )
    answer = tokenizer.decode(out[0][input_ids.shape[1]:]).replace(text, '').replace('\n', '')
    return answer


def chat(weight_path):
    assert weight_path is not None
    net.load_state_dict(torch.load(f'{weight_path}/adapter_model.bin'), strict=False)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    history = []
    with torch.no_grad():
        while True:
            text = input('User: ')
            text = meta_instruction + text
            history.append(text)
            answer = inference('\n'.join(history[-5:]))
            history.append(answer)
            print(f'XiaoRuan: {answer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default=None)
    args = parser.parse_args()
    chat(args.weight_path)
