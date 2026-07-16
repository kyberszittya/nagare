import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root/"reports/figures/hsikan_pytorch_bench.json"))
ts = d["thread_scaling_50k_ms"]; mt = d["matched_16thr"]
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(15.5,4.5))
# panel 1: thread scaling (50k)
th=[int(k) for k in ts]; ms=[ts[str(k)] for k in th]
ideal=[ms[0]/t for t in th]
a1.plot(th,ms,"o-",color="#2a9d5a",lw=2.6,label="Nagare HSiKAN (measured)")
a1.plot(th,ideal,"--",color="#9fb8a8",lw=1.5,label="ideal linear")
a1.axhline(51.3,color="#c0392b",ls=":",lw=1.8,label="PyTorch best (6–16 thr)")
for t,m in zip(th,ms): a1.annotate(f"{ms[0]/m:.1f}×",(t,m),textcoords="offset points",xytext=(3,6),fontsize=8,color="#2a7d4f")
a1.set_xscale("log",base=2); a1.set_xticks(th); a1.set_xticklabels(th)
a1.set_xlabel("rayon threads"); a1.set_ylabel("fwd+bwd ms/iter (50k edges)")
a1.set_title("Nagare HSiKAN now scales with threads\n7.3× at 16 threads"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
# panel 2: matched 16-thread time
e=[f"{r['edges']//1000}k" for r in mt]; x=np.arange(len(e)); w=0.36
a2.bar(x-w/2,[r["nagare_ms"] for r in mt],w,color="#2a9d5a",label="Nagare (16 thr)")
a2.bar(x+w/2,[r["pytorch_ms"] for r in mt],w,color="#c0392b",label="PyTorch (16 thr)")
for i,r in enumerate(mt): a2.annotate(f"{r['pytorch_ms']/r['nagare_ms']:.1f}× faster",(i,r["nagare_ms"]),textcoords="offset points",xytext=(0,4),ha="center",fontsize=8,color="#2a7d4f")
a2.set_xticks(x); a2.set_xticklabels(e); a2.set_ylabel("ms/iter"); a2.set_xlabel("hyperedges")
a2.set_title("Matched 16 threads: time"); a2.legend(fontsize=8); a2.grid(alpha=0.3,axis="y")
# panel 3: matched 16-thread memory
a3.bar(x-w/2,[r["nagare_rss_mb"] for r in mt],w,color="#2a9d5a",label="Nagare (16 thr)")
a3.bar(x+w/2,[r["pytorch_rss_mb"] for r in mt],w,color="#c0392b",label="PyTorch (16 thr)")
for i,r in enumerate(mt): a3.annotate(f"{r['pytorch_rss_mb']/r['nagare_rss_mb']:.1f}× less",(i,r["nagare_rss_mb"]),textcoords="offset points",xytext=(0,4),ha="center",fontsize=8,color="#2a7d4f")
a3.set_xticks(x); a3.set_xticklabels(e); a3.set_ylabel("peak RSS (MB)"); a3.set_xlabel("hyperedges")
a3.set_title("Matched 16 threads: memory"); a3.legend(fontsize=8); a3.grid(alpha=0.3,axis="y")
fig.suptitle("HSiKAN parallelized (edge-chunk rayon): scales to 7.3×, and at matched 16 threads beats PyTorch ~5× on time + ~3× on memory",fontsize=11.5,y=1.03)
fig.tight_layout(); out=root/"reports/figures/hsikan-bench.png"; fig.savefig(out,dpi=140,bbox_inches="tight"); print("wrote",out)
