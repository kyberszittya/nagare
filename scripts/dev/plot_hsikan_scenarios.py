import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root/"reports/figures/hsikan_pytorch_bench.json"))
k = d["kato15_32core"]; A = d["scenario_A_dim_sweep"]; B = d["scenario_B_deploy_vs_train"]
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(16,4.6))
# panel 1: kato15 32-core scaling, Nagare graceful vs PyTorch thrash
nn=k["nagare_fwdbwd_ms"]["50k"]; pp=k["pytorch_fwdbwd_ms"]["50k"]
th=[1,2,4,8,16,32]; nms=[nn[str(t)] for t in th]
pth=[1,8,32]; pms=[pp[str(t)] for t in pth]
a1.plot(th,nms,"o-",color="#2a9d5a",lw=2.6,label="Nagare (graceful → 32ms)")
a1.plot(pth,pms,"s-",color="#c0392b",lw=2.2,label="PyTorch (thrash → 5360ms @32)")
a1.annotate("33× regression\n(over-subscription)",(32,5360),textcoords="offset points",xytext=(-96,-6),fontsize=8,color="#c0392b")
a1.set_yscale("log"); a1.set_xscale("log",base=2); a1.set_xticks(th); a1.set_xticklabels(th)
a1.set_xlabel("threads"); a1.set_ylabel("fwd+bwd ms/iter (log)")
a1.set_title("kato15 32-core: Nagare degrades gracefully,\nPyTorch collapses at 32 threads"); a1.legend(fontsize=8); a1.grid(alpha=0.3,which="both")
# panel 2: dim-sweep scaling (flat)
dims=["d16","d32","d64"]; sc=[A["scaling_1to16"][x] for x in dims]; xx=np.arange(3)
a2.bar(xx,sc,0.55,color=["#2a9d5a","#3aa d6a".replace(" ",""),"#4abd7a"])
a2.axhline(16,color="#9fb8a8",ls="--",lw=1.2,label="ideal (16 thr)")
for i,v in enumerate(sc): a2.annotate(f"{v}×",(i,v),textcoords="offset points",xytext=(0,4),ha="center",fontsize=9,color="#2a7d4f")
a2.set_xticks(xx); a2.set_xticklabels(["d=16","d=32","d=64"]); a2.set_ylim(0,17)
a2.set_ylabel("1→16-thread speedup"); a2.set_title("Scenario A: scaling ~flat in d (7.3–7.9×)\n→ not d-bandwidth-bound below 16 threads"); a2.legend(fontsize=8); a2.grid(alpha=0.3,axis="y")
# panel 3: deploy vs train (time + memory as grouped)
labels=["fwd-only\n(deploy)","fwd+bwd\n(train)"]; xx=np.arange(2); w=0.36
nt=[B["nagare"]["fwd_only_ms"],B["nagare"]["fwd_bwd_ms"]]; pt=[B["pytorch"]["fwd_only_ms"],B["pytorch"]["fwd_bwd_ms"]]
a3.bar(xx-w/2,nt,w,color="#2a9d5a",label="Nagare ms")
a3.bar(xx+w/2,pt,w,color="#c0392b",label="PyTorch ms")
a3.set_xticks(xx); a3.set_xticklabels(labels); a3.set_ylabel("ms/iter")
nr=[B["nagare"]["fwd_only_rss_mb"],B["nagare"]["fwd_bwd_rss_mb"]]; pr=[B["pytorch"]["fwd_only_rss_mb"],B["pytorch"]["fwd_bwd_rss_mb"]]
a3.annotate(f"RSS: {nr[0]}MB vs {pr[0]}MB\n= 29× less (deploy)",(0,nt[0]),textcoords="offset points",xytext=(-6,22),fontsize=8,color="#2a7d4f")
a3.annotate(f"RSS: {nr[1]}MB vs {pr[1]}MB",(1,pt[1]),textcoords="offset points",xytext=(-30,4),fontsize=8,color="#555")
a3.set_title("Scenario B: deploy vs train (50k, d16, 8thr)\nno-tape → backward +29% (PyTorch +150%)"); a3.legend(fontsize=8); a3.grid(alpha=0.3,axis="y")
fig.suptitle("HSiKAN further scenarios: 32-core robustness · dim-scaling (flat) · deploy-vs-train (16MB deploy footprint)",fontsize=11.5,y=1.03)
fig.tight_layout(); out=root/"reports/figures/hsikan-scenarios.png"; fig.savefig(out,dpi=140,bbox_inches="tight"); print("wrote",out)
