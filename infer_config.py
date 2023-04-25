import torch.cuda as cuda

device = 'cuda:0' if cuda.is_available() else 'cpu'
max_new_tokens = 768
temperature = 0.25
top_p = 0.7
repetition_penalty = 1.02


