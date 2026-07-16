import sys, time, torch
def argv(flag, default, cast=int):
    for a in sys.argv:
        if a.startswith(flag+"="): return cast(a.split("=")[1])
    return default
D  = argv("--dim", 16)
T  = argv("--edges", 50000)
CB, ITERS = 6, 100
FWD_ONLY = any(a=="--fwd-only" for a in sys.argv)
N = max(T//2, 64)
g = torch.Generator().manual_seed(0)
x  = torch.empty(N, D).uniform_(-1,1,generator=g).requires_grad_(not FWD_ONLY)
vert = torch.randint(0, N, (T,3), generator=g)
sign = (torch.randint(0,2,(T,3),generator=g).float()*2-1)
inner = torch.empty(2,D,CB).uniform_(-.3,.3,generator=g).requires_grad_(not FWD_ONLY)
outer = torch.empty(2,D,CB).uniform_(-.3,.3,generator=g).requires_grad_(not FWD_ONLY)
Wt = torch.empty(D,D).uniform_(-.2,.2,generator=g).requires_grad_(not FWD_ONLY)
bt = torch.full((D,), -1.0).requires_grad_(not FWD_ONLY)
gt = torch.empty(T, D).uniform_(-.01,.01,generator=g)
def cheb(z, coef):
    Tk=[torch.ones_like(z), z]
    for k in range(2,CB): Tk.append(2*z*Tk[-1]-Tk[-2])
    return (torch.stack(Tk,-1)*coef).sum(-1)
def fwd():
    hv = x[vert]
    gate = torch.sigmoid(hv @ Wt + bt)
    sp = (sign>0).unsqueeze(-1).float()
    inner_val = sp*cheb(hv, inner[0]) + (1-sp)*cheb(hv, inner[1])
    mixed = gate*inner_val + (1-gate)*hv
    cp = sp.sum(1).clamp(min=1); cn = (1-sp).sum(1).clamp(min=1)
    aggp = (mixed*sp).sum(1)/cp; aggn = (mixed*(1-sp)).sum(1)/cn
    return cheb(aggp, outer[0]) + cheb(aggn, outer[1])
def step():
    if FWD_ONLY:
        with torch.no_grad(): fwd()
    else:
        (fwd()*gt).sum().backward()
        for p in (x,inner,outer,Wt,bt): p.grad=None
step()
t0=time.perf_counter()
for _ in range(ITERS): step()
ms=(time.perf_counter()-t0)*1e3/ITERS
mode = "fwd-only" if FWD_ONLY else "fwd+bwd"
print(f"PyTorch HSiKAN edges={T} d={D} {mode} {ms:.3f} ms/iter ({ITERS} iters) threads={torch.get_num_threads()}")
