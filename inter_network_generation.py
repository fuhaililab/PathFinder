import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ujson as json
from math import sqrt
import pandas as pd
import data_utils
from constants import *



def generate_inter_network(receptor_save_dir, ligands, fcs, save_dir):
    network_database = np.load("data/network/processed_network.npz", allow_pickle=True)
    lr_network = network_database["lr_network"]

    with open(f"{receptor_save_dir}/intra_network.json") as f:
        graph_data = json.load(f)
    receptor_intra_network = nx.node_link_graph(graph_data)
    receptors = set()
    network_edges = list(receptor_intra_network.edges)
    for edge in network_edges:
        edge_type = receptor_intra_network.get_edge_data(edge[0], edge[1])["edge_type"]
        if edge_type in [0, 2, 5]:
            receptors.add(edge[0])
    receptors = list(receptors)

    lr_pairs = set()
    for ligand, fc in zip(ligands, fcs):
        keep_pair = lr_network[lr_network[:, 0] == ligand, :]
        if len(keep_pair) == 0:
            continue
        else:
            for receptor in receptors:
                if receptor in keep_pair[:, 1]:
                    lr_pairs.add((ligand, receptor, fc))
    lr_pairs = pd.DataFrame(list(lr_pairs))
    lr_pairs.columns = ["source", "target", "ligand_abs_fc"]
    lr_pairs.to_csv(f"{save_dir}/inter_network.txt", index=None, sep='\t')




if __name__ == "__main__":
    network_database = np.load("data/network/processed_network.npz", allow_pickle=True)
    lr_network = network_database["lr_network"]
    all_ligands = lr_network[:, 0]

    # mic-ex
    ligand_DEGs = pd.read_csv("data/ad_mice/DEGs/mic.csv")
    keep_index = [True if gene in all_ligands else False for gene in ligand_DEGs.iloc[:, 0]]
    ligands_DEGs = ligand_DEGs.iloc[keep_index, :]
    ligands = ligands_DEGs.loc[ligand_DEGs["p_val"] < 0.05, "Unnamed: 0"].tolist()
    fcs = np.abs(ligands_DEGs.loc[ligand_DEGs["p_val"] < 0.05, "avg_log2FC"]).tolist()

    generate_inter_network("save/train_classifier/ex-01", ligands, fcs, "generated_networks/mic_ex")

    # ast-ex / ast-mic
    ligand_DEGs = pd.read_csv("data/ad_mice/DEGs/ast.csv")
    keep_index = [True if gene in all_ligands else False for gene in ligand_DEGs.iloc[:, 0]]
    ligands_DEGs = ligand_DEGs.iloc[keep_index, :]
    ligands = ligands_DEGs.loc[ligand_DEGs["p_val"] < 0.05, "Unnamed: 0"].tolist()
    fcs = np.abs(ligands_DEGs.loc[ligand_DEGs["p_val"] < 0.05, "avg_log2FC"]).tolist()

    generate_inter_network("save/train_classifier/ex-01", ligands, fcs, "generated_networks/ast_ex")
    generate_inter_network("save/train_classifier/mic-01", ligands, fcs, "generated_networks/ast_mic")

    # ex_ast / ex_mic
    ligand_DEGs = pd.read_csv("data/ad_mice/DEGs/ex.csv")
    keep_index = [True if gene in all_ligands else False for gene in ligand_DEGs.iloc[:, 0]]
    ligands_DEGs = ligand_DEGs.iloc[keep_index, :]
    ligands = ligands_DEGs.loc[ligand_DEGs["p_val"] < 0.05, "Unnamed: 0"].tolist()
    fcs = np.abs(ligands_DEGs.loc[ligand_DEGs["p_val"] < 0.05, "avg_log2FC"]).tolist()

    generate_inter_network("save/train_classifier/ast-03", ligands, fcs, "generated_networks/ex_ast")
    generate_inter_network("save/train_classifier/mic-01", ligands, fcs, "generated_networks/ex_mic")




