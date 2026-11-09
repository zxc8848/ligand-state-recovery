"""Current-method vector figures; never reads predictions or changes frozen results."""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY = ROOT/'work/evaluation_fix_v3/development/unified_resonance_summary.json'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})


def overview():
    # Keep the manuscript overview in a small independent editable vector script.
    from draw_fig1 import overview as draw_overview
    draw_overview()


def regimes():
    data = json.loads(SUMMARY.read_text(encoding='utf-8'))['conditions']
    archived = json.loads((ROOT/'experiments/archived_graph_tables_20261102/SUMMARY.json').read_text())['conditions']
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for ax, prefix, xs, title in [(axes[0], 'pdbbind', [0,.05,.10,.15], 'Deposited poses (n=488)'),
                                 (axes[1], 'ccd', [0,.10], 'CCD ideal coordinates (n=1000)')]:
        keys = [f'{prefix}_sigma{x:.2f}'.replace('.', 'p') for x in xs]
        rows = [data[key] for key in keys]
        ax.errorbar(xs, [st.mean(r['new_per_seed']) for r in rows],
                    yerr=[st.stdev(r['new_per_seed']) for r in rows],
                    fmt='o-', capsize=3, color='#246f9b', label='Current decoder (mean +/- SD)')
        ax.plot(xs, [r['knodle'] for r in rows], 's--', color='#bd6b28', label='Knodle')
        ax.plot(xs, [archived[key]['full']['unified_resonance'] for key in keys],
                '^-.', color='#756A8C', linewidth=1.6, label='Archived raw MILP')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Coordinate noise sigma (angstrom)')
        ax.set_xticks(xs); ax.set_ylim(0, 1.02); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Resonance-aware exact recovery')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5,-.03), fontsize=8.5)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(HERE/'figures/fig6_current_regimes.pdf', bbox_inches='tight')
    fig.savefig(HERE/'figures/fig6_current_regimes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    overview(); regimes()
    print('Generated current-method overview and paired regime figures from', SUMMARY)
