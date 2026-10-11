"""Regenerate the local halo plot and audit diagram from archived estimates."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Paper13B/latex/figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'savefig.facecolor':'white', 'figure.facecolor':'white'})
data = pd.read_csv(ROOT / 'Paper13B/upgrade/structure_audit.csv')
models = ['qwen2.5:7b-instruct', 'qwen2.5:14b-instruct', 'gemma2:27b', 'qwen2.5:32b-instruct-q4_K_M']
fig, ax = plt.subplots(figsize=(8.5, 4.8))
fig.subplots_adjust(left=.11, right=.97, bottom=.25, top=.87)
x = np.arange(4)
for condition, offset, color in [('joint',-.18,'#0072B2'), ('contrastive',.18,'#D55E00')]:
    y = [data[(data.method == model+'/'+condition) & (data.corpus=='pooled')].iloc[0].halo_pairwise for model in models]
    ax.bar(x+offset, y, width=.33, color=color, label=condition.capitalize(), zorder=3)
    for xx, yy in zip(x+offset, y):
        ax.annotate(f'{yy:.3f}', (xx,yy), xytext=(0,7),
                    textcoords='offset points', ha='center', color=color, fontsize=10)
ax.set_xticks(x, ['Qwen 7B','Qwen 14B','Gemma 27B','Qwen 32B'])
ax.set_xlim(-.5,3.5)
ax.set_ylim(0,1)
ax.set_ylabel('Mean cross-construct Pearson $r$ (halo)')
ax.grid(axis='y', alpha=.22)
ax.legend(loc='lower center', bbox_to_anchor=(.5,1.02), ncol=2, frameon=False)
fig.text(.11,.08,'300 source documents per condition; quantized local checkpoints, one seed.\nMeans use condition-specific supports and 7–10 estimable pairs; not paired prompt effects.',
         fontsize=9, color='#444444', va='bottom')
fig.savefig(OUT/'f12_local_scaling_halo.png', dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.2,7))
fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
def box(cx, cy, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle='round,pad=0.012,rounding_size=0.014',
                               facecolor=color,edgecolor='#617285',linewidth=1))
    ax.text(cx,cy+h/2-.027,title,ha='center',va='top',fontsize=11,fontweight='bold',color='#16324B')
    ax.text(cx,cy-.016,body,ha='center',va='center',fontsize=10,linespacing=1.45,color='#24384A')
def arrow(a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':'#617285','lw':1.4,'shrinkA':3,'shrinkB':3})
box(.5,.90,.88,.13,'Fixed corpus and codebook',
    'Amazon • Yelp • Regulations.gov • Bluesky\nATT • PE • EE • SI • BI; absence retained as missing level', '#EAF2F8')
box(.17,.66,.27,.22,'Study 1: diagnosis',
    '4,800-document benchmark\n1,600-document nested core\nQwen / Mistral; contrast arms\nSeeds, prompt variants,\nperturbations and languages','#EAF2F8')
box(.50,.66,.27,.22,'Study 2: elicitation',
    '1,000 documents (250/corpus)\nJoint baseline vs isolated,\nevidence and presence gate\nOriginal Qwen; one seed', '#FFF2E5')
box(.83,.66,.27,.22,'Local extension',
    '300 documents (75/corpus)\nJoint vs contrastive prompt\n7B / 14B / 27B / 32B\n2,400 calls; one seed', '#EDF5EF')
for cx in [.17,.50,.83]: arrow((cx,.82),(cx,.78))
box(.5,.36,.88,.20,'Diagnostics on explicit observation supports',
    'Cross-construct correlations + pair denominators + coverage\nPairwise-complete correlation PCA vs complete-case PCA\nGaussian one-factor ML diagnostic; document-bootstrap intervals\nMatched-support prompt comparisons; stability in original repeated runs','#F2F3F6')
for cx in [.17,.50,.83]: arrow((cx,.54),(cx,.47))
box(.5,.12,.88,.13,'Study 3 and interpretation boundary',
    'PC1 removal; held-out global-rating prediction\nOutput structure ≠ semantic accuracy or an identified representation-level cause','#F2F3F6')
arrow((.5,.25),(.5,.20))
fig.savefig(OUT/'f13_methodological_audit_pipeline.png', dpi=300)
plt.close(fig)
print('Generated local halo plot and audit diagram in',OUT)
