import torch

data = [[1, 2], [3, 4]]
ts1 = torch.tensor(data)

if(torch.cuda.is_available()):
    ts1.to("cuda")


ts2 = torch.cat([ts1, ts1, ts1], dim=0)
print(ts2)