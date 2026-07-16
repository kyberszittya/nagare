import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root/"reports/figures/hsikan_pytorch_bench.json"))["rows"]
e = [r["edges"] for r in d]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4))
a1.plot(e,[r["nagare_ms"] for r in d],"o-",color="#2a9d5a",lw=2.4,label="Nagare (1 thread)")
a1.plot(e,[r["pytorch_ms_1thr"] for r in d],"s-",color="#e08a2a",lw=2,label="PyTorch (1 thread)")
a1.plot(e,[r["pytorch_ms_6thr"] for r in d],"^--",color="#c99a5a",lw=1.8,label="PyTorch (6 threads)")
a1.set_xlabel("hyperedges"); a1.set_ylabel("fwd+bwd ms/iter"); a1.set_title("Time: Nagare 2.3× faster per-core\n(PyTorch wins wall-clock only via 6 threads)")
a1.legend(fontsize=8); a1.grid(alpha=0.3)
a2.plot(e,[r["nagare_rss_mb"] for r in d],"o-",color="#2a9d5a",lw=2.4,label="Nagare (no autograd tape)")
a2.plot(e,[r["pytorch_rss_mb"] for r in d],"s-",color="#c0392b",lw=2.2,label="PyTorch (autograd tape)")
for r in d: a2.annotate(f"{r['pytorch_rss_mb']/r['nagare_rss_mb']:.1f}×",(r["edges"],r["nagare_rss_mb"]),textcoords="offset points",xytext=(4,-12),fontsize=8,color="#2a7d4f")
a2.set_xlabel("hyperedges"); a2.set_ylabel("peak RSS (MB)"); a2.set_title("Memory: Nagare 4–5× less peak RSS")
a2.legend(fontsize=8); a2.grid(alpha=0.3)
fig.suptitle("HSiKAN fwd+bwd, Nagare vs PyTorch on CPU (same architecture): per-core 2.3× faster + 4–5× less memory",fontsize=11,y=1.03)
fig.tight_layout(); out=root/"reports/figures/hsikan-bench.png"; fig.savefig(out,dpi=140,bbox_inches="tight"); print("wrote",out)
