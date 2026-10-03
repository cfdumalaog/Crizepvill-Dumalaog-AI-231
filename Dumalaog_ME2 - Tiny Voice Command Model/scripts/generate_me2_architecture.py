"""Draw verified stage connections for the two scratch TinyDSCNN-48 models."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tinyvcm_model.model import TinyDSCNN  # noqa: E402

OUT = PROJECT / 'docs'
OUT.mkdir(exist_ok=True)


def model_stages(classes: int):
    model = TinyDSCNN(classes=classes).eval()
    x = torch.zeros(1, 1, 40, 251)
    stages = [('Input log-mel', '(1, 40, 251)', 0)]
    boundaries = [(0, 3, '5x5 stem / BN / ReLU'),
                  (3, 9, 'DS block 1'), (9, 15, 'DS block 2'),
                  (15, 21, 'DS block 3'), (21, 27, 'DS block 4')]
    with torch.inference_mode():
        for first, last, name in boundaries:
            block = model.features[first:last]
            x = block(x)
            params = sum(p.numel() for p in block.parameters())
            stages.append((name, str(tuple(x.shape[1:])), params))
        x = model.pool(x).flatten(1)
    stages.append(('Global average pool', str(tuple(x.shape[1:])), 0))
    stages.append((f'Linear head: {classes} logits', f'({classes},)', sum(p.numel() for p in model.classifier.parameters())))
    assert sum(row[2] for row in stages) == sum(p.numel() for p in model.parameters())
    return stages


def draw_graph(nodes, edges, positions, path, title, width=17, height=5):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor('#ffffff')
    ax.set_title(title, loc='left', fontsize=17, fontweight='bold', color='#102a43', pad=22)
    nx.draw_networkx_edges(graph, positions, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=19,
                           node_size=9000, width=1.7, edge_color='#4474a5', min_source_margin=16, min_target_margin=16)
    colors = ['#dcecfb' if i < len(nodes) - 2 else '#d9f4e8' for i in range(len(nodes))]
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, edgecolors='#2c5b87',
                           linewidths=1.4, node_shape='s', node_size=9000)
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=10, font_family='DejaVu Sans',
                            font_color='#102a43', font_weight='normal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    intent = model_stages(31)
    wake = model_stages(2)
    meta = {'intent_stages': intent, 'wake_stages': wake,
            'intent_parameters': sum(s[2] for s in intent), 'wake_parameters': sum(s[2] for s in wake)}
    (OUT / 'ME2_ARCHITECTURE_VERIFIED.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    labels = []
    for name, shape, params in intent:
        display_name = (name.replace('5x5 stem / BN / ReLU', '5x5 stem' + chr(10) + 'BN / ReLU')
                        .replace('Global average pool', 'Global average' + chr(10) + 'pool')
                        .replace('Linear head: ', 'Linear head' + chr(10)))
        labels.append(f'{display_name}\n{shape}\n{params:,} params')
    positions = {label: (0, -i * 4.5) for i, label in enumerate(labels)}
    draw_graph(labels, list(zip(labels, labels[1:])), positions, OUT / 'ME2_TinyDSCNN48_connections.pdf',
               'TinyDSCNN-48 layer connections (31-class intent head)', width=8, height=13)
    nodes = ['16 kHz microphone', '2.5 s ring buffer', '40 x 251 log-mel',
             'Binary wake\nNON_WAKE / WAKE_WORD', 'Wake threshold',
             '31-class intent\nTinyDSCNN-48', 'Coded action\n+ UI / GPIO']
    positions = {node: (0, -i * 4.5) for i, node in enumerate(nodes)}
    draw_graph(nodes, list(zip(nodes, nodes[1:])), positions, OUT / 'ME2_two_stage_connections.pdf',
               'Always-listening two-model speech-to-action path', width=8, height=12)
    print(json.dumps({k: v for k, v in meta.items() if k.endswith('parameters')}))


if __name__ == '__main__':
    main()
