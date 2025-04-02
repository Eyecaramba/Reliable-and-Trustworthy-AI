import torch
import torch.nn as nn

# Targeted FGSM Attack
def fgsm_targeted(model, x, target, eps):
    x.requires_grad = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = x.grad.data
    perturbed_data = x - eps * data_grad.sign()
    return torch.clamp(perturbed_data, 0, 1)

# Untargeted FGSM Attack
def fgsm_untargeted(model, x, label, eps):
    x.requires_grad = True
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()
    data_grad = x.grad.data
    perturbed_data = x + eps * data_grad.sign()
    return torch.clamp(perturbed_data, 0, 1)

# Targeted PGD Attack
def pgd_targeted(model, x, target, k, eps, eps_step):
    perturbed_data = x.clone().detach()
    perturbed_data.requires_grad = True
    for _ in range(k):
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data - eps_step * data_grad.sign()
        perturbed_data = torch.max(torch.min(perturbed_data, x + eps), x - eps)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        perturbed_data = perturbed_data.detach()
        perturbed_data.requires_grad = True
    return perturbed_data

# Untargeted PGD Attack
def pgd_untargeted(model, x, label, k, eps, eps_step):
    perturbed_data = x.clone().detach()
    perturbed_data.requires_grad = True
    for _ in range(k):
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_data.grad.data
        perturbed_data = perturbed_data + eps_step * data_grad.sign()
        perturbed_data = torch.max(torch.min(perturbed_data, x + eps), x - eps)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        perturbed_data = perturbed_data.detach()
        perturbed_data.requires_grad = True
    return perturbed_data