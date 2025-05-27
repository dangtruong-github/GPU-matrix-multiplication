import torch
import naive

A = torch.randn(2, 2, device='cuda')
B = torch.randn(2, 2, device='cuda')
result = naive.matmul(A, B)
print("CUDA Kernel Output:", result)
