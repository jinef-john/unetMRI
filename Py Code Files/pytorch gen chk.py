import torch
print(torch.__version__)
print(torch.__file__)
print(torch.randn_like(torch.zeros(3, 3), generator=torch.Generator())
)
