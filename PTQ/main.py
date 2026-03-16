
import os
import torch
import torch.nn as nn
from Preprocess import datapool
from utils import train, val, seed_all, get_logger

from NonNormedModel import get_norm_model
from bias_folding_utils import fuse_model, bias_folding_SNN_trainable, train_bias_weights


get_ipython().system('pip uninstall opencv-python opencv-contrib-python')
get_ipython().system('pip install opencv-python-headless')


args = {
    'workers': 4,
    'batch_size': 200,
    'seed': 42,
    'suffix': '',
    'dataset': 'cifar10',
    'model': 'resnet20',
    'identifier': 'resnet20_L[16]_normed_on_th_added_act_wo_lin_act',  
    'device': '0',
    'time': 0
}


train_loader, val_loader = datapool(args['dataset'], args['batch_size'])

os.environ["CUDA_VISIBLE_DEVICES"] = args['device']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_norm_pooling = get_norm_model(args)
model_norm_pooling.to(device)

model_norm_pooling.set_T(0)
model_norm_pooling.set_L(16)
acc = val(model_norm_pooling, val_loader, device, 0)
print(f"Validation accuracy ann baseline: {acc}")

model_norm_pooling.set_T(16)
model_norm_pooling.set_L(16)
acc = val(model_norm_pooling, val_loader, device, 16)
print(f"Validation accuracy snn baseline: {acc}")

model_norm_pooling.set_T(0)
fused_model = fuse_model(model_norm_pooling)
fused_model.to(device)
acc = val(fused_model, val_loader, device, 0)
print(f"Validation accuracy fused ann baseline: {acc}")

fused_model.set_T(16)
fused_model.set_L(16)
model_snn = bias_folding_SNN_trainable(fused_model)

acc = val(model_snn, val_loader, device, 16)
print(f"Validation accuracy fused snn baseline: {acc}")

model = train_bias_weights(
    model=model_snn,
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=10,
    lr=0.0001,
    device='cuda'
)

