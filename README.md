# Awesome Chinese Chatbots
To try all models. Run:
```bash
git clone https://github.com/SchweitzerGAO/awesome-chinese-chatbot
```
## ChatGLM
A chatbot with PTM [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) with instruction fine-tuning and parameter-freezing by [PEFT](https://github.com/huggingface/peft)
### Acknowledgement
I would like to extend my sincerest appreciation to the developers of 
- [ChatGLM](https://github.com/THUDM/ChatGLM-6B)
- [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
- [ChatGLM-LLaMA-chinese-instruct](https://github.com/27182812/ChatGLM-LLaMA-chinese-insturct)

With the help of these LLM weight and code, I made a fine-tuned LLM of myself.

To process the steps below, run:
```bash
cd chatglm-6b
```
### Finetune
To finetune, run:
```bash
./finetune.sh
```
You can modify the parameters in `finetune.sh`

To see what parameters you can modify, run:
```bash
python finetune.py -h
```

### Chat
To chat with the bot, run
```bash
python infer.py --weight_path PATH_TO_YOUR_WEIGHT
```

## MOSS
Coming Soon...
