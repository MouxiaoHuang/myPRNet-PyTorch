import numpy as np
from ResFCN256 import ResFCN256

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

model = ResFCN256()

print('The number of model params: ', params_count(model))