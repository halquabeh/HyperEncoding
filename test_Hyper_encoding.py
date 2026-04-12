import torch
B = 2  
T = 4
inputs = torch.rand(B, 3, 32, 32)
v_t = inputs * T # B,C,H,W
out = []
for t in range(T):
    s_t = torch.bernoulli(1/(T-t+1)*v_t)
    v_t = torch.clamp(v_t - s_t, min=0)
    out.append(s_t)
out = torch.stack(out, dim=0) # T,B,C,H,W
out = out.flatten(0, 1).contiguous() #TxB,C,H,W : this is a convention in the code to run smoothing in CNN
print(out.shape)
