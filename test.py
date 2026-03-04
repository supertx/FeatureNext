import torch

print(torch.load("checkpoints/palm_finetune/palm_finetuned_final.pt", weights_only=True).keys())