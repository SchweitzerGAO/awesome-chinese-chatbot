from typing import Optional

from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
from transformers.trainer import TRAINING_ARGS_NAME
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
net = AutoModel.from_pretrained("THUDM/chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map="auto")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Cast2Float(nn.Sequential):
    def forward(self, X):
        return super().forward(X).to(torch.float32)


def prepare_model(model, finetune_args):
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = Cast2Float(model.lm_head)
    model.config.use_cache = False
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h", 'lm_head']
    )
    model = get_peft_model(model, peft_config)
    return model


@dataclass
class FineTuneArguments:
    data_path: str = field(default='./data')
    lora_rank: int = field(default=8)


def collate_fn(features):
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ChatGLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(input_ids=inputs['input_ids'], labels=inputs['labels']).loss

    def save_model(self, output_dir: Optional[str] = './saved_models', _internal_call: bool = False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to('cpu') for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, 'adapter_model.bin'))


def finetune():
    global net
    writer = SummaryWriter()
    training_args, finetune_args = HfArgumentParser((TrainingArguments,
                                                     FineTuneArguments)
                                                    ).parse_args_into_dataclasses()
    net = prepare_model(net, finetune_args)
    dataset = datasets.load_from_disk(finetune_args.data_path)
    trainer = ChatGLMTrainer(
        model=net,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=collate_fn
    )
    trainer.train()
    writer.close()
    net.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    finetune()
