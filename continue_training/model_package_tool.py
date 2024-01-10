import torch

model = torch.load("./continue_training_model/model.pt")


# model = torch.load('model.pt')
def traverse_params(module):
    for name, param in module.items():
        if isinstance(param, torch.nn.Module):
            traverse_params(param)
        else:
            print(name, param.size())

traverse_params(model)


# print(model)
