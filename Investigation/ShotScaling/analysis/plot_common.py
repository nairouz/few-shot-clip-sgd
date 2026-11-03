from pathlib import Path
import matplotlib.pyplot as plt
def shot_plot(frame,y,out,title):
 fig,ax=plt.subplots(figsize=(6,4))
 for name,g in frame.groupby("dataset"):ax.plot(g["shots"],g[y],marker="o",label=name)
 ax.set_xscale("log",base=2);ax.set_xlabel("Shots (K)");ax.set_ylabel(y);ax.set_title(title);ax.legend(fontsize=7)
 fig.tight_layout();Path(out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=300);fig.savefig(Path(out).with_suffix(".pdf"));plt.close(fig)
