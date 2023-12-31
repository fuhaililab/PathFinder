import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ujson as json
from math import sqrt

import data_utils

from constants import *


def generate_intra_network(save_dir, keep_path=50, output_dir=None):
    result = np.load(f"{save_dir}/save_result.npz", allow_pickle=True)
    path_weights = result["path_weight"]
    path_weight = np.mean(path_weights, axis=0)
    print(path_weight)
    print(np.max(path_weight))
    print(np.min(path_weight))
    path_list = result["path_list"]
    path_index = result["path_index"]
    gene_list = result["gene_list"]
    receptor_list = result["receptor_list"]
    with open(f"{save_dir}/original_graph.json") as f:
        G_data = json.load(f)
    G = nx.node_link_graph(G_data)

    # order path from high to low
    path_order = np.argsort(path_weight)[::-1]
    keep_path = min(keep_path, path_order.shape[0])
    keep_path_list = path_order[:keep_path]

    new_G = nx.DiGraph()
    edge_list = []
    for index in keep_path_list:
        path = path_list[path_index == index]
        path_length = path.shape[0]
        for i in range(path_length - 1):
            source = gene_list[path[i]]
            target = gene_list[path[i + 1]]
            edge_list.append((source, target, {"edge_type": G.get_edge_data(source, target)["edge_type"]}))
    new_G.add_edges_from(edge_list)

    if output_dir is not None:
        plot_intra_network(new_G, f"{output_dir}/intra_network.png")
        G_data = nx.node_link_data(new_G)
        data_utils.save_json(f"{output_dir}/intra_network.json", G_data)
    else:
        plot_intra_network(new_G, f"{save_dir}/intra_network.png")
        G_data = nx.node_link_data(new_G)
        data_utils.save_json(f"{save_dir}/intra_network.json", G_data)


def plot_intra_network(G, file_path):
    plt.figure(figsize=(16, 9))
    edge_attr = np.array(list(nx.get_edge_attributes(G, "edge_type").values()))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=20)
    labels = nx.draw_networkx_labels(G, pos, font_size=8)
    for edge_type, edge_color in zip(EDGE_LIST, EDGE_COLOR):
        nx.draw_networkx_edges(G, pos, width=1.5,
                               edgelist=tuple(np.array(G.edges())[edge_attr == EDGE_DICT[edge_type]]),
                               label=edge_type, edge_color=edge_color)
    from matplotlib.lines import Line2D
    def make_proxy(clr, **kwargs):
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)

    proxies = [make_proxy(clr, lw=5) for clr in EDGE_COLOR]
    plt.legend(proxies, list(EDGE_DICT.keys()))
    plt.savefig(file_path)
    plt.close()

