import torch.cuda as cuda

device = 'cuda:0' if cuda.is_available() else 'cpu'
max_new_tokens = 512
temperature = 0
top_p = 0.95
repetition_penalty = 1.02


