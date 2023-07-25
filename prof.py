import torch
from pytorch_memlab import LineProfiler
from pytorch_memlab import profile_every

@profile_every(1)
def inner():
    torch.nn.Linear(100, 100).cuda()

def outer():
    linear = torch.nn.Linear(100, 100).cuda()
    linear2 = torch.nn.Linear(100, 100).cuda()
    inner()

outer()

# with LineProfiler(outer, inner) as prof:
#     outer()
# prof.display()

# z = torch.tensor([[1, 2, 3], [4, 5, 6]])
# x = torch.tensor([[4, 5, 3], [6, 7, 4], [8, 9, 5], [10, 11, 6]])
# print(z)
# print(x)

# result = z * x[..., None, :]
# print(result)
# print(f"z.shape {z.shape}, x.shape {x.shape}, result.shape {result.shape}")

# p = result.sum(dim=-1)
# print(p)
# print(f"p.shape {p.shape}")
