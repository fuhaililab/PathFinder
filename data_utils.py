"""
Data utils files
"""


import random
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import torch
import ujson as json
from networkx import Graph
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter
import pandas as pd
from sklearn.preprocessing import scale
from constants import *




def reindex_nx_graph(G: Graph, ordered_node_list: list) -> Graph:
    r"""reindex the nodes in nx graph according to given ordering.
    Args:
        G (Graph): Networkx graph object.
        ordered_node_list (list): A list served as node ordering.
    """

    ordered_node_dict = dict(zip(ordered_node_list, range(len(ordered_node_list))))
    return nx.relabel_nodes(G, ordered_node_dict)


def save_json(filename: str,
              obj: dict,
              message: Optional[str] = None,
              ascii: Optional[bool] = True):
    r"""Save data in JSON format.
    Args:
        filename (str) : Name of save directory (including file name).
        obj (object): Data to be saved.
        message (Optional, str): Anything to print.
        ascii (Optional, bool): If ture, ensure the encoding is ascii.
    """
    if message is not None:
        print(f"Saving {message}...")
    with open(filename, "w") as fh:
        json.dump(obj, fh, ensure_ascii=ascii)


def nx_to_graph_data(G: Graph, num_nodes: int ) -> Data:
    r"""convert nx graph to torch geometric Data object
    Args:
        G（nx.Graph): Networkx graph.
        num_nodes(Tensor): a sclar tensor to save the number of node in the graph
    """
    edge_list = G.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).to(torch.long)
    in_deg = torch.tensor([node for (node, val) in sorted(G.in_degree, key=lambda pair: pair[0])]).long().unsqueeze(-1)
    out_deg = torch.tensor([node for (node, val) in sorted(G.out_degree, key=lambda pair: pair[0])]).long().unsqueeze(
        -1)
    return Data(edge_index=edge_index, num_nodes=num_nodes, in_deg=in_deg, out_deg=out_deg)


def nx_compute_in_and_out_degree(G: Graph) -> Tuple[Tensor, Tensor]:
    r"""Compute in and out degree of each node in the input graph.
    Args:
        G (nx.Graph): Networkx graph.
    """
    in_deg = torch.tensor([val for (node, val) in sorted(G.in_degree, key=lambda pair: pair[0])]).int().unsqueeze(-1)
    out_deg = torch.tensor([val for (node, val) in sorted(G.out_degree, key=lambda pair: pair[0])]).int().unsqueeze(-1)
    return in_deg, out_deg


def nx_compute_shortest_path(G: Graph,
                             min_length: int,
                             max_length: int,
                             num_edges: int,
                             keep_gene_list: Optional[list] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    r"""Compute all pair shortest path in the graph.
    Args:
        G (nx.Graph): Networkx graph.
        min_length (int): Minimum length of the path.
        max_length (int): Maximum length considered when computing the shortest path.
        keep_gene_list (Optional, list): If specified, only keep path that start from a gene in the list.
    """
    shortest_path_pair = nx.all_pairs_shortest_path(G, max_length)
    all_path_list = []
    path_index = []
    path_edge_type = []
    path_position = []
    path_count = 0
    for shortest_path in shortest_path_pair:
        index, paths = shortest_path
        if keep_gene_list is not None:
            if index not in keep_gene_list:
                continue

        for end_node, path in paths.items():
            if end_node == index:
                continue
            elif len(path) < min_length and G.get_edge_data(path[-2], path[-1])["edge_type"] != 5:
                continue
            else:
                path_edges = []
                for i in range(len(path) - 1):
                    path_edges.append(G.get_edge_data(path[i], path[i + 1])["edge_type"])
                # add padding edge type
                path_edges.append(num_edges)

                path_edge_type.extend(path_edges)
                path_position.extend([i for i in range(len(path))])
                all_path_list.extend(path)
                path_index.extend([path_count for _ in range(len(path))])
                path_count += 1

    return torch.tensor(all_path_list).long(), \
           torch.tensor(path_index).long(), \
           torch.tensor(path_edge_type).long(), \
           torch.tensor(path_position).long(),\
           path_count


def nx_compute_all_simple_paths(G: Graph,
                                source_node_list: list,
                                target_node_list: list,
                                min_length: int,
                                max_length: int,
                                num_edges: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    r"""Compute all possible paths between the source node and target node in the list. The path must follow the rule
        receptor -> target / receptor -> tf -> target / receptor -> sig -> tf -> target.
    Args:
        G (nx.Graph): Networkx graph.
        source_node_list (list): List of the source node.
        target_node_list (list): List of the target node.
        min_length (int): Minimum length of the path.
        max_length (int): Maximum length of the path.
        num_edges (int): Number of edge types in the graph.
    """
    all_path_list = []
    path_index = []
    path_edge_type = []
    path_position = []
    path_count = 0
    count = 0
    for source in source_node_list:
        for target in target_node_list:
            path_list = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
            for path in path_list:
                if len(path) >= min_length:
                    if G.get_edge_data(path[-2], path[-1])["edge_type"] in [4, 5]:
                        path_edges = []
                        for i in range(len(path) - 1):
                            path_edges.append(G.get_edge_data(path[i], path[i + 1])["edge_type"])
                        # add padding edge type
                        path_edges.append(num_edges)
                        all_path_list.extend(path)
                        path_index.extend([path_count for _ in range(len(path))])
                        path_edge_type.extend(path_edges)
                        path_position.extend([i for i in range(len(path))])
                        path_count += 1
            count += 1

    return torch.tensor(all_path_list).long(), \
           torch.tensor(path_index).long(), \
           torch.tensor(path_edge_type).long(), \
           torch.tensor(path_position).long(),\
           path_count


def nx_combine_shortest_path_and_simple_path(G: Graph,
                                             source_node_list: list,
                                             target_node_list: list,
                                             min_length: int,
                                             max_length: int,
                                             num_edges: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    r"""Compute and combin both the shortest path and
        all possible simple paths between the source node and target node in the list.
    Args:
        G (nx.Graph): Networkx graph.
        source_node_list (list): List of the source node.
        target_node_list (list): List of the target node.
        min_length (int): Minimum length of the path.
        max_length (int): Maximum length of the path.
    """
    shortest_path_list, shortest_path_index, shortest_path_type, shortest_path_positions, shortest_path_count\
        = nx_compute_shortest_path(G, min_length, max_length, num_edges, source_node_list)
    simple_path_list, simple_path_index, simple_path_type, simple_path_positions, simple_path_count = \
        nx_compute_all_simple_paths(G, source_node_list, target_node_list, min_length, max_length, num_edges)

    total_path_list = torch.cat([shortest_path_list, simple_path_list], dim=0)
    total_path_index = torch.cat([shortest_path_index, simple_path_index + shortest_path_count], dim=0)
    total_path_type = torch.cat([shortest_path_type, simple_path_type], dim=0)
    total_path_positions = torch.cat([shortest_path_positions, simple_path_positions], dim=0)
    total_path_count = simple_path_count + shortest_path_count
    # total_path_list = []
    # total_path_index = []
    # total_path_type = []
    # total_path_positions = []
    # total_path_count = shortest_path_count
    # for i in range(shortest_path_count):
    #     path = shortest_path_list[shortest_path_index == i].numpy().tolist()
    #     total_path_list.append(path)
    #     total_path_index.append([i for _ in range(len(path))])
    #
    # for i in range(simple_path_count):
    #     path = simple_path_list[simple_path_index == i].numpy().tolist()
    #     if path not in total_path_list:
    #         total_path_list.append(path)
    #         total_path_index.append([total_path_count for _ in range(len(path))])
    #         total_path_count += 1

    print(total_path_count)
    return total_path_list, \
           total_path_index, \
           total_path_type, \
           total_path_positions,\
           total_path_count




def nx_compute_all_random_path(G: Graph,
                               min_length: int,
                               max_length: int) -> Tensor:
    r"""Random select one path for each pair of the node if there exist.
    Args:
        G (nx.Graph): Networkx graph.
        min_length (int): Minimum length when computing path.
        max_length (int): Maximum length when computing path.
    """
    all_path_list = []
    path_index = []
    path_count = 0
    graph_nodes = list(G.nodes)
    for source in graph_nodes:
        for target in graph_nodes:
            path_list = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
            if len(path_list) > 0:
                pair_path_index = [i for i in range(len(path_list))]
                while len(pair_path_index) > 0:
                    random.shuffle(pair_path_index)
                    index = pair_path_index.pop()
                    path = path_list[index]
                    if len(path) < min_length:
                        continue
                    else:
                        all_path_list.extend(path)
                        path_index.extend([path_count for _ in range(len(path))])
                        path_count += 1
                        break
    return torch.tensor(all_path_list, dtype=torch.long), torch.tensor(path_index, dtype=torch.long), path_count


def get_path_prior_weight(fold_change: Tensor,
                          path_list: LongTensor,
                          path_index: LongTensor,
                          mode: str = "up") -> Tensor:
    r"""Compute the prior weight for each path based on the fold-change value of all genes in the path.

    Args:
        fold_change (Tensor): Fold-change value for each gene.
        path_list (LongTensor): Path gene index.
        path_index (LongTensor): Path index.
        mode (str, optional): prior weight mode, choose from (up, down, deg).


    """
    if mode == "down":
        fold_change = -fold_change
    elif mode == "deg":
        fold_change = torch.abs(fold_change)

    weight = fold_change[path_list].view(-1, 1)
    total_weight = scatter(weight, path_index, dim=-2, reduce="mean")
    normalized_weight = (total_weight - total_weight.min()) / (total_weight.max() - total_weight.min())
    return normalized_weight


def nx_compute_shortest_path_length(G: Graph,
                                    max_length: int) -> Tensor:
    r"""Compute all pair the shortest path length in the graph.
    Args:
        G (nx.Graph): Networkx graph.
        max_length (int): Maximum length when computing the shortest path.
    """
    num_node = G.number_of_nodes()
    shortest_path_length_matrix = torch.zeros([num_node, num_node]).int()
    all_shortest_path_lengths = nx.all_pairs_shortest_path_length(G, max_length)
    for shortest_path_lengths in all_shortest_path_lengths:
        index, path_lengths = shortest_path_lengths
        for end_node, path_length in path_lengths.items():
            if end_node == index:
                continue
            else:
                shortest_path_length_matrix[index, end_node] = path_length
    return shortest_path_length_matrix


def nx_return_edge_feature_index(G: Graph) -> Tensor:
    r"""Return edge type for each edge in the graph.
    Args:
        G (nx.Graph): Networkx graph.
    """
    return torch.from_numpy(nx.adjacency_matrix(G, weight="edge_type").toarray()).long()

def torch_from_json(path: str, dtype: Optional[torch.dtype]=torch.float32) -> Tensor:
    r"""Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor

def process_input_data(args):
    # Load training data.
    control_X = pd.read_csv(args.control_file_path, sep="\s+", header=None).values.T
    test_X = pd.read_csv(args.test_file_path, sep="\s+", header=None).values.T
    input_gene_list = np.load(args.gene_symbol_file_path, allow_pickle=True)["gene_list"]


    # Only keep top genes in the data.
    control_X = control_X[:, :args.top_gene]
    test_X = test_X[:, :args.top_gene]
    input_gene_list = input_gene_list[:args.top_gene]

    # Create dataset and labels
    expression = np.concatenate([control_X, test_X], axis=0)
    num_control = control_X.shape[0]
    num_test = test_X.shape[0]
    label = np.array([0 for _ in range(expression.shape[0])])
    label[num_control:] = 1

    # remove duplicated genes from dataset
    unique, counts = np.unique(input_gene_list, return_counts=True)
    duplicated_gene_list = unique[np.where(counts > 1)[0]].tolist()
    unique_gene_index = np.array([i for i, gene in enumerate(input_gene_list) if gene not in duplicated_gene_list])
    keep_expression = np.zeros([expression.shape[0], len(duplicated_gene_list)])
    for i, gene in enumerate(duplicated_gene_list):
        index = np.where(input_gene_list == gene)[0]
        keep_index = index[np.argmax(np.var(expression[:, index], axis=0))]
        keep_expression[:, i] = expression[:, keep_index]
    expression = expression[:, unique_gene_index]
    expression = np.concatenate([expression, keep_expression], axis=-1)
    input_gene_list = input_gene_list[unique_gene_index]
    input_gene_list = np.concatenate([input_gene_list, duplicated_gene_list])

    # Split back to control and test
    control_X = expression[:num_control]
    test_X = expression[num_control:]

    # filter genes with low expression or low express percentages
    control_expr_pre = np.sum(control_X > 0, axis=0) / control_X.shape[0]
    control_expr_mean = np.mean(control_X, axis=0)
    control_expr_filter = np.logical_and(control_expr_pre > 0.05, control_expr_mean > 0.05)

    test_expr_pre = np.sum(test_X > 0, axis=0) / test_X.shape[0]
    test_expr_mean = np.mean(test_X, axis=0)
    test_expr_filter = np.logical_and(test_expr_pre > 0.05, test_expr_mean > 0.05)

    # compute log-fold-change
    fold_change = np.log(np.mean(test_X, axis=0) + EPS) - np.log(np.mean(control_X, axis=0) + EPS)
    fold_change[np.logical_or(np.logical_not(control_expr_filter), np.logical_not(test_expr_filter))] = 0.0

    if args.add_coexp:
    # get co-expression network
        corr = np.corrcoef(expression.T)
        corr = np.nan_to_num(corr, 0)
        np.fill_diagonal(corr, 0)
        expressed_gene = np.logical_and(control_expr_filter, test_expr_filter)[:, np.newaxis] * 1
        express_matrix = expressed_gene * expressed_gene.T
        co_expression_network = np.logical_and(corr > 0.3, express_matrix == 1) * 1
        # co-expression network
        co_exp_network = []
        row, col = np.where(co_expression_network == 1)
        for row_index, col_index in zip(row, col):
            co_exp_network.append([input_gene_list[row_index], input_gene_list[col_index], "co_expression"])
    else:
        co_exp_network = []
    co_exp_network.append([None, None, None])

    expression = scale(expression, axis=0)


    network_database = np.load("data/network/processed_network.npz", allow_pickle=True)
    sig_network = network_database["sig_network"]
    gr_network = network_database["gr_network"]
    lr_network = network_database["lr_network"]
    receptor_list = np.unique(lr_network[:, 1])

    # collate gene in signaling network and gene regulatory network database
    intra_network = np.concatenate([sig_network, gr_network, co_exp_network], axis=0)
    intra_network = intra_network[:-1]
    input_gene_set = set(input_gene_list)
    keep_gene_set = set()
    for i in range(intra_network.shape[0]):
        source = intra_network[i, 0]
        target = intra_network[i, 1]
        if source in input_gene_set and target in input_gene_set:
            keep_gene_set.add(source)
            keep_gene_set.add(target)

    if len(keep_gene_set) < 100:
        raise ValueError("The number of genes in database is too small for computation...")

    keep_gene_index = [True if gene in keep_gene_set else False for gene in input_gene_list]
    # reconstruct gene expression data
    expression = expression[:, keep_gene_index]
    keep_gene_list = input_gene_list[keep_gene_index]
    fold_change = fold_change[keep_gene_index]
    keep_receptor_index = [True if gene in keep_gene_set else False for gene in receptor_list]
    receptor_list = receptor_list[keep_receptor_index]

    # reconstruct network
    sig_sub_network = []
    for i in range(sig_network.shape[0]):
        source = sig_network[i, 0]
        target = sig_network[i, 1]
        edge_type = sig_network[i, 2]
        if source in keep_gene_list and target in keep_gene_list:
            sig_sub_network.append([source, target, edge_type])

    gr_sub_network = []
    target_set = set()
    for i in range(gr_network.shape[0]):
        source = gr_network[i, 0]
        target = gr_network[i, 1]
        edge_type = gr_network[i, 2]
        if source in keep_gene_list and target in keep_gene_list:
            gr_sub_network.append([source, target, edge_type])
            target_set.add(target)
    target_list = list(target_set)

    # for intra network, to keep original signaling flow, use directed graph
    G = nx.DiGraph()
    G.add_nodes_from(keep_gene_list)
    for source, target, edge_type in sig_sub_network:
        G.add_edge(source, target, edge_type=EDGE_DICT[edge_type])
    for source, target, edge_type in gr_sub_network:
        G.add_edge(source, target, edge_type=EDGE_DICT[edge_type])

    if args.add_coexp:
        for source, target, edge_type in co_exp_network:
            if not G.has_edge(source, target):
                G.add_edge(source, target, edge_type=EDGE_DICT[edge_type])

    # save original graph
    G_data = nx.node_link_data(G)
    save_json(f"{args.save_dir}/original_graph.json", G_data)

    # process graph
    G = reindex_nx_graph(G, keep_gene_list)

    edge_types = nx_return_edge_feature_index(G)
    args.num_edges = len(np.unique(edge_types))

    receptor_list = np.array([i if gene in receptor_list else None for i, gene in enumerate(keep_gene_list)])
    receptor_list = receptor_list[receptor_list != None]
    target_list = np.array([i if gene in target_list else None for i, gene in enumerate(keep_gene_list)])
    target_list = target_list[target_list != None]
    args.num_nodes = len(keep_gene_list)


    path_list, path_index, path_edge_type, path_positions, args.num_paths = nx_combine_shortest_path_and_simple_path(G, receptor_list,
                                                target_list, 3, args.max_length, args.num_edges)

    prior_path_weight = get_path_prior_weight(torch.tensor(fold_change).float(), path_list, path_index, args.reg_mode)
    shortest_path_length = nx_compute_shortest_path_length(G, args.max_length)
    in_deg, out_deg = nx_compute_in_and_out_degree(G)
    args.num_in_degree = torch.max(in_deg).item() + 1
    args.num_out_degree = torch.max(out_deg).item() + 1


    return args, \
           expression, \
           edge_types, \
           G, \
           keep_gene_list, \
           label, \
           receptor_list, \
           target_list, \
           fold_change, \
           prior_path_weight, \
           shortest_path_length, \
           in_deg, \
           out_deg, \
           path_list, \
           path_index, \
           path_edge_type, \
           path_positions


