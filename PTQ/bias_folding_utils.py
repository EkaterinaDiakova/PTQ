import os
import torch
import torch.nn as nn
import Models

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def fuse_conv_bn(conv, bn):
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    ).to(conv.weight.device)

    fused_conv.weight.data = conv.weight.data.clone()

    if conv.bias is not None:
        fused_conv.bias.data = conv.bias.data.clone()
    else:
        fused_conv.bias.data = torch.zeros(conv.out_channels).to(conv.weight.device)

    bn_mean = bn.running_mean
    bn_var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    bn_weight = bn.weight
    bn_bias = bn.bias


    scale_factor = bn_weight / bn_var_sqrt

    fused_conv.weight.data = fused_conv.weight.data * scale_factor.reshape(-1, 1, 1, 1)

    fused_conv.bias.data = (fused_conv.bias.data - bn_mean) * scale_factor + bn_bias

    return fused_conv

def fuse_all_conv_bn_pairs(module, parent=None, name=None):

    for child_name, child in module.named_children():
        fuse_all_conv_bn_pairs(child, module, child_name)

    if isinstance(module, nn.Sequential):
        modules_to_replace = []


        for i in range(len(module)):
            if isinstance(module[i], nn.BatchNorm2d):
                if i > 0 and isinstance(module[i-1], nn.Conv2d):
                    modules_to_replace.append((i-1, i, module[i-1], module[i]))

        for conv_idx, bn_idx, conv, bn in reversed(modules_to_replace):
            fused_conv = fuse_conv_bn(conv, bn)

            new_modules = []
            for j in range(len(module)):
                if j == conv_idx:
                    new_modules.append(fused_conv)
                elif j == bn_idx:
                    continue
                else:
                    new_modules.append(module[j])


            module.__init__(*new_modules)

def fuse_model(model):
    fused_model = copy.deepcopy(model)

    fuse_all_conv_bn_pairs(fused_model)

    return fused_model

def bias_folding_SNN_trainable(model):
    model_copy = copy.deepcopy(model)

    original_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
    bias_cache = {k: v.clone().detach() for k, v in original_state_dict.items() if '.bias' in k}

    print("Bias cache keys:", list(bias_cache.keys()))

    bias_mapping = {
        'conv1.1': ['conv1.0.bias'],
        'conv2_x.0.residual_function.1': ['conv2_x.0.residual_function.0.bias'],
        'conv2_x.0.act': ['conv2_x.0.residual_function.2.bias'],
        'conv2_x.1.residual_function.1': ['conv2_x.1.residual_function.0.bias'],
        'conv2_x.1.act': ['conv2_x.1.residual_function.2.bias'],
        'conv2_x.2.residual_function.1': ['conv2_x.2.residual_function.0.bias'],
        'conv2_x.2.act': ['conv2_x.2.residual_function.2.bias'],
        'conv3_x.0.residual_function.1': ['conv3_x.0.residual_function.0.bias'],
        'conv3_x.0.act': ['conv3_x.0.residual_function.2.bias', 'conv3_x.0.shortcut.0.bias'],
        'conv3_x.1.residual_function.1': ['conv3_x.1.residual_function.0.bias'],
        'conv3_x.1.act': ['conv3_x.1.residual_function.2.bias'],
        'conv3_x.2.residual_function.1': ['conv3_x.2.residual_function.0.bias'],
        'conv3_x.2.act': ['conv3_x.2.residual_function.2.bias'],
        'conv4_x.0.residual_function.1': ['conv4_x.0.residual_function.0.bias'],
        'conv4_x.0.act': ['conv4_x.0.residual_function.2.bias', 'conv4_x.0.shortcut.0.bias'],
        'conv4_x.1.residual_function.1': ['conv4_x.1.residual_function.0.bias'],
        'conv4_x.1.act': ['conv4_x.1.residual_function.2.bias'],
        'conv4_x.2.residual_function.1': ['conv4_x.2.residual_function.0.bias'],
        'conv4_x.2.act': ['conv4_x.2.residual_function.2.bias'],
        'avg_pool_act': [],
    }

    def create_bias_decay_weights(T):
        """Создает логарифмические веса для затухания bias"""
        decay_weights = torch.zeros(T)
        for t in range(T):
            weight = (math.log(T + 1) - math.log(t + 1)) / math.log(T + 1)
            decay_weights[t] = weight
        sum_weights = decay_weights.sum()
        decay_weights = decay_weights / sum_weights
        return nn.Parameter(decay_weights)

    for if_name, bias_names in bias_mapping.items():
        module = None
        for name, mod in model_copy.named_modules():
            if name == if_name:
                module = mod
                break

        if module is not None and hasattr(module, 'thresh'):
            print(f"\nProcessing module: {if_name}")
            print(f"  Module type: {type(module)}")
            print(f"  Has T attribute: {hasattr(module, 'T')}")
            if hasattr(module, 'T'):
                print(f"  T = {module.T}")

            found_bias = []
            for bias_name in bias_names:
                if bias_name in bias_cache:
                    found_bias.append(bias_name)
                else:
                    print(f"  Warning: Bias {bias_name} not found in cache")


            if hasattr(module, 'T') and module.T > 0:
                if found_bias:
                    bias_weights = create_bias_decay_weights(module.T)
                    module.register_parameter('bias_decay_weights', bias_weights)
                    print(f"  ✓ Registered bias_decay_weights (T={module.T}) for biases: {found_bias}")
                else:
                    print(f"  ✗ No biases found for {if_name}, skipping bias_decay_weights registration")

            patched_forward = create_patched_forward(bias_names, bias_cache, if_name)
            module.forward = patched_forward.__get__(module, type(module))
            print(f"  Patched forward method for {if_name}")

    with torch.no_grad():
        all_used_bias = set()
        for bias_names in bias_mapping.values():
            all_used_bias.update(bias_names)

        print("\nZeroing biases:")
        for bias_name in all_used_bias:
            if bias_name in model_copy.state_dict():
                print(f"  Zeroing: {bias_name}")
                model_copy.state_dict()[bias_name].zero_()
            else:
                print(f"  Warning: Bias {bias_name} not found in state_dict")

    print("\nFinal state_dict keys with 'bias_decay_weights':")
    for key in model_copy.state_dict().keys():
        if 'bias_decay_weights' in key:
            print(f"  {key}")

    return model_copy

def create_patched_forward(bias_list, bias_cache_dict, module_name):
    def patched_forward(self, x):
        total_bias = None
        for bias_name in bias_list:
            if bias_name in bias_cache_dict:
                bias_tensor = bias_cache_dict[bias_name].to(x.device)
                if total_bias is None:
                    total_bias = bias_tensor.clone()
                elif total_bias.shape == bias_tensor.shape:
                    total_bias += bias_tensor
                else:
                    print(f"Warning: Shape mismatch for {bias_name} in {module_name}")

        if self.T > 0 and total_bias is not None:
            if not hasattr(self, 'bias_decay_weights'):
                print(f"Warning: {module_name} has no bias_decay_weights, but has bias!")
                return x

            thre = self.thresh.data
            decay_weights = self.bias_decay_weights

            x_expanded = self.expand(x)
            cur_sample = x_expanded[0]

            mem = torch.zeros_like(cur_sample)
            mem += 0.5 * thre.expand_as(mem)

            spike_pot = []
            for t in range(self.T):
                cur_input = x_expanded[t]

                bias_weighted = total_bias * decay_weights[t]
                if len(mem.shape) == 4:
                    bias_expanded = bias_weighted.view(1, -1, 1, 1).expand_as(mem)
                else:
                    bias_expanded = bias_weighted.view(1, -1).expand_as(mem)

                mem = mem + cur_input + bias_expanded

                spike = self.act(mem - thre.expand_as(mem), self.gama)
                mem = mem - spike * thre.expand_as(mem)
                spike_pot.append(spike)

            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        elif self.T > 0 and total_bias is None:
            # Нет bias для этого модуля - работаем как обычно
            pass

        return x
    return patched_forward

from tqdm import tqdm

def train_bias_weights(model, train_loader, test_loader, epochs=5, lr=0.0001, device='cuda'):
    """
    Простая функция для обучения только bias_decay_weights

    Args:
        model: модель после bias_folding_SNN_trainable
        train_loader: DataLoader для обучения
        test_loader: DataLoader для тестирования
        epochs: количество эпох
        lr: learning rate
        device: устройство
    """
    # Замораживаем все параметры, кроме bias_decay_weights
    print("Обучаемые параметры:")
    trainable_params = []
    for name, param in model.named_parameters():
        if 'bias_decay_weights' in name:
            param.requires_grad = True
            trainable_params.append(name)
            print(f"  ✓ {name}")
        else:
            param.requires_grad = False

    print(f"\nВсего обучаемых параметров: {len(trainable_params)}")
    print("-" * 50)

    # Оптимизатор только для обучаемых параметров
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if len(outputs.shape) == 3:
                outputs = outputs.mean(dim=0)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*predicted.eq(targets).sum().item()/targets.size(0):.2f}%'
            })

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Тестирование
        model.eval()
        test_correct = 0
        test_total = 0

        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False)
        with torch.no_grad():
            for inputs, targets in test_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                if len(outputs.shape) == 3:
                    outputs = outputs.mean(dim=0)

                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                test_pbar.set_postfix({
                    'acc': f'{100.*predicted.eq(targets).sum().item()/targets.size(0):.2f}%'
                })

        test_acc = 100. * test_correct / test_total

        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        print("-" * 50)

    return model

#train_loader, val_loader = datapool(args['dataset'], args['batch_size'])

# os.environ["CUDA_VISIBLE_DEVICES"] = args['device']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_norm_pooling = get_norm_model(args)
# model_norm_pooling.to(device)

# model_norm_pooling.set_T(0)
# model_norm_pooling.set_L(16)
# acc = val(model_norm_pooling, val_loader, device, 0)
# print(f"Validation accuracy ann baseline: {acc}")

# model_norm_pooling.set_T(16)
# model_norm_pooling.set_L(16)
# acc = val(model_norm_pooling, val_loader, device, 16)
# print(f"Validation accuracy snn baseline: {acc}")

# fused_model = fuse_model(model_norm_pooling)
# fused_model.to(device)

# fused_model.set_T(0)
# fused_model.set_L(16)
# acc = val(fused_model, val_loader, device, 0)
# print(f"Validation accuracy fused ann baseline: {acc}")

# fused_model.set_T(16)
# fused_model.set_L(16)
# model_snn = bias_folding_SNN_trainable(fused_model)

# acc = val(model_snn, val_loader, device, 16)
# print(f"Validation accuracy fused snn baseline: {acc}")

# model = train_bias_weights(
#     model=model_snn,
#     train_loader=train_loader,
#     test_loader=val_loader,
#     epochs=10,
#     lr=0.0001,
#     device='cuda'
# )

