import sys, time, torch, torch.nn.functional as F
D, CB, ITERS = 16, 6, 100
T = int([a.split('=')[1] for a in sys.argv if a.startswith('--edges=')][0]) if any('--edges=' in a for a in sys.argv) else 50000
N = max(T//2, 64)
g = torch.Generator().manual_seed(0)
x  = torch.empty(N, D).uniform_(-1,1,generator=g).requires_grad_(True)
vert = torch.randint(0, N, (T,3), generator=g)
sign = (torch.randint(0,2,(T,3),generator=g).float()*2-1)          # ±1
inner = torch.empty(2,D,CB).uniform_(-.3,.3,generator=g).requires_grad_(True)
outer = torch.empty(2,D,CB).uniform_(-.3,.3,generator=g).requires_grad_(True)
Wt = torch.empty(D,D).uniform_(-.2,.2,generator=g).requires_grad_(True)
bt = torch.full((D,), -1.0).requires_grad_(True)
gt = torch.empty(T, D).uniform_(-.01,.01,generator=g)             # upstream grad target
def cheb(z, coef):                                                # z(...,D) coef(D,CB)->(...,D)
    Tk=[torch.ones_like(z), z]
    for k in range(2,CB): Tk.append(2*z*Tk[-1]-Tk[-2])
    return (torch.stack(Tk,-1)*coef).sum(-1)
def step():
    hv = x[vert]                                                  # (T,3,D)
    gate = torch.sigmoid(hv @ Wt + bt)
    sp = (sign>0).unsqueeze(-1).float()
    inner_val = sp*cheb(hv, inner[0]) + (1-sp)*cheb(hv, inner[1])
    mixed = gate*inner_val + (1-gate)*hv
    cp = sp.sum(1).clamp(min=1); cn = (1-sp).sum(1).clamp(min=1)
    aggp = (mixed*sp).sum(1)/cp; aggn = (mixed*(1-sp)).sum(1)/cn
    he = cheb(aggp, outer[0]) + cheb(aggn, outer[1])              # (T,D)
    (he*gt).sum().backward()
    for p in (x,inner,outer,Wt,bt): p.grad=None
step()  # warmup
t0=time.perf_counter()
for _ in range(ITERS): step()
ms=(time.perf_counter()-t0)*1e3/ITERS
print(f"PyTorch HSiKAN edges={T} d={D} fwd+bwd {ms:.3f} ms/iter ({ITERS} iters) threads={torch.get_num_threads()}")
