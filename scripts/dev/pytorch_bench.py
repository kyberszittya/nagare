import time, sys, numpy as np, torch, torch.nn as nn
torch.manual_seed(0)
def load(name):
    a = np.loadtxt(f"/tmp/nb_{name}.csv", delimiter=",", dtype=np.float32)
    return a[:, :-1], a[:, -1]
def mlp(din, dout):
    return nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Linear(64, dout))
def train_time(din, dout, xtr, ytr, cls):
    def once():
        torch.manual_seed(0)
        m = mlp(din, dout); opt = torch.optim.Adam(m.parameters(), lr=0.01)
        lossf = nn.CrossEntropyLoss() if cls else nn.MSELoss()
        Xt = torch.tensor(xtr); Yt = torch.tensor(ytr).long() if cls else torch.tensor(ytr).view(-1,1)
        t0 = time.perf_counter()
        for _ in range(300):
            opt.zero_grad(); out = m(Xt); loss = lossf(out, Yt); loss.backward(); opt.step()
        return (time.perf_counter()-t0)*1e3, m
    ms = sorted(once()[0] for _ in range(3))[1]
    return ms, once()[1]
def evalm(m, xte, yte, cls):
    with torch.no_grad():
        out = m(torch.tensor(xte))
        if cls: return (out.argmax(1).numpy() == yte.astype(int)).mean()
        p = out.view(-1).numpy(); ss = ((yte-yte.mean())**2).sum(); return 1 - ((yte-p)**2).sum()/ss
print(f"PyTorch {torch.__version__} — MLP d->64->out, 300 epochs, Adam. threads: {torch.get_num_threads()}")
print(f"{'dataset':<12}{'metric':>9}{'train_ms':>11}")
for name, fp, din, dout, cls in [("iris","iris",4,3,True), ("california","cali",8,1,False)]:
    xtr,ytr = load(f"{fp}_train"); xte,yte = load(f"{fp}_test")
    ms, m = train_time(din, dout, xtr, ytr, cls)
    print(f"{name:<12}{evalm(m,xte,yte,cls):>9.3f}{ms:>10.1f}")
