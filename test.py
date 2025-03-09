import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.device(1))
print(torch.cuda.get_device_name(1))
print(torch.__version__)