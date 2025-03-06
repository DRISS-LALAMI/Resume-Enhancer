import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Check if GPU is available

device = torch.device("cpu")
print(f"Using device: {device}")
