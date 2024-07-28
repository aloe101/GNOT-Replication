import torch.nn.functional as F

def pad_to_max_length(tensor, max_length):
    return F.pad(tensor, (0, 0, 0, max_length - tensor.shape[0]))
