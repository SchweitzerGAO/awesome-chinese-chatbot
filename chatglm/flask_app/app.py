from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
from peft import PeftModel
import infer_config


app = Flask(__name__)
CORS(app)

meta_instruction = '你的名字是小软，是基于开源语言模型在党史问答数据集上微调的党史问答机器人，你可以与用户闲聊，并回答与党史相关的问题。\n'

weight_path = '../saved_models/epoch_8'
torch.set_default_tensor_type(torch.cuda.HalfTensor)
net = AutoModel.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True, load_in_8bit=False, device_map='auto').half()
tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
net = PeftModel.from_pretrained(net, weight_path)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

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


@app.route('/text_chat', methods=['POST'])
def text_chat():
    query = request.json['message']
    ans = inference(query)
    return jsonify({'code': 200, 'data': [ans]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
