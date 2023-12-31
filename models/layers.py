"""

Pathfinder layer construction

"""
from math import sqrt
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from torch_scatter import scatter, scatter_softmax

from .utils import clones


def attention(query: Tensor,
              key: Tensor,
              value: Tensor,
              bias: Tensor,
              mask: Tensor,
              dropout: Optional[nn.Dropout] = None) -> Tensor:
    r"""Self attention function with masking.
    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        bias (torch.Tensor): Bias value for attention score.
        mask (torch.Tensor): Mask tensor to indicate the mask position. 0 for mask.
        dropout (nn.Dropout): Dropout layer.
    """
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-1, -2).contiguous()) / sqrt(d_k)  # Batch_size * h * seq_len * seq_len
    score = score + bias
    score = score.masked_fill_(mask == 0, -1e30)
    attn = F.softmax(score, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn  # Batch_size * h * seq_len * d_k


class MultiHeadedAttention(nn.Module):
    r"""
    Multi-Header attention mechanism.
    Args:
        hidden_size (int): Hidden size of model.
        h (int): Number of head in multi-head attention.
        drop_prob (Optional, float): Dropout probability.
    """

    def __init__(self,
                 hidden_size: int,
                 h: int,
                 drop_prob: Optional[float] = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % h == 0
        self.hidden_size = hidden_size
        self.h = h
        self.attn = None
        self.linears = clones(nn.Linear(hidden_size, hidden_size), 3)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                bias: Tensor,
                mask: Tensor) -> Tensor:
        batch_size = query.size(0)
        mask = mask.view(batch_size, 1, 1, -1)
        d_k = self.hidden_size // self.h
        query, key, value = [l(x).view(batch_size, -1, self.h, d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.linears)]  # Batch_size * h * seq_len * d_k

        x, self.attn = attention(query, key, value, mask, bias)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * d_k)  # Batch_size * seq_len * hidden_size
        return x


class PointwiseFeedForwardNetwork(nn.Module):
    r"""Feed forward NN.
    Args:
        hidden_size (int): hidden size of the model.
        drop_prob (Optional, float): dropout probability.
    """

    def __init__(self,
                 hidden_size: int,
                 drop_prob: Optional[float] = 0.1):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SublayerConnection(nn.Module):
    r"""A residual connection followed by a layer norm.
    Args:
        hidden_size (int): Hidden size of the model.
        drop_prob (Optional, float): Dropout probability.
    """

    def __init__(self,
                 hidden_size: int,
                 drop_prob: Optional[float]):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        r"Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    r"""Encoder is made up of self-attn and feed forward (defined below).
    Args:
        hidden_size (int): Hidden size of the model.
        self_attn (nn.Module): Multi-head attention module.
        feed_forward (nn.Module): Feed forward network.
        path_embedding_layer (nn.Module): Path embedding layer.
        drop_prob (Optional, float): Dropout probability.
    """

    def __init__(self,
                 hidden_size: int,
                 self_attn: nn.Module,
                 feed_forward: nn.Module,
                 path_embedding_layer: nn.Module,
                 drop_prob: Optional[float] = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(hidden_size, drop_prob), 2)
        self.path_embedding_layer = path_embedding_layer
        self.hidden_size = hidden_size

    def forward(self,
                x: Tensor,
                bias: Tensor,
                mask: Tensor,
                path_list: list,
                path_index: LongTensor,
                path_edge_type: LongTensor,
                path_positions: LongTensor,
                path_weight: Tensor) -> Tuple[Tensor, Tensor]:
        r"Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, bias, mask))
        x = self.sublayer[1](x, self.feed_forward)
        path_emb = self.path_embedding_layer(x, path_list, path_index, path_edge_type, path_positions, path_weight)
        return x, path_emb


class EncoderBlock(nn.Module):
    """Encoder block, consist of multiple encoder layer.
    Args:
        layer (nn.Module): Encoder layer.
        N (int): Number of layer.
    """

    def __init__(self, layer: nn.Module, N: int):
        super(EncoderBlock, self).__init__()
        self.layer_list = clones(layer, N)
        self.norm = nn.LayerNorm(layer.hidden_size)

    def forward(self,
                x: Tensor,
                bias: Tensor,
                mask: Tensor,
                path_list: list,
                path_index: LongTensor,
                path_edge_type: LongTensor,
                path_positions: LongTensor,
                path_weight: Tensor) -> Tuple[Tensor, List[Tensor]]:
        path_emb_list = []
        for l in self.layer_list:
            x, path_emb = l(x, bias, mask, path_list, path_index, path_edge_type, path_positions, path_weight)
            path_emb_list.append(path_emb)
        return self.norm(x), path_emb_list


class PathEmbedding(nn.Module):
    r"""Path embedding and path weight learning.
    Args:
        hidden_size (int): Hidden size of the model.
        r (int): Fixed embedding length for paths with arbitrary length.
        max_path_len (int): Maximum length of input paths.
        num_edges (int): Number of edge types.
        drop_prob (Optional, float): Dropout probability.
    """

    def __init__(self,
                 hidden_size: int,
                 r: int,
                 max_path_len: int,
                 num_edges: int,
                 drop_prob: Optional[float] = 0.1):
        super(PathEmbedding, self).__init__()
        assert hidden_size % r == 0
        self.hidden_size = hidden_size
        self.r = r
        self.max_path_len = max_path_len
        self.num_edges = num_edges
        self.norm = nn.LayerNorm(hidden_size)
        emb_size = hidden_size // r
        self.path_edge_emb = nn.Embedding(num_edges + 1, emb_size)
        self.path_positional_emb = nn.Embedding(max_path_len, emb_size)
        self.path_proj = nn.Linear(hidden_size, emb_size)
        self.aggregate_proj = nn.Sequential(nn.Linear(emb_size, r), nn.Tanh(), nn.Dropout(drop_prob), nn.Linear(r, r))

    def forward(self,
                x: Tensor,
                path_list: LongTensor,
                path_index: LongTensor,
                path_edge_type: LongTensor,
                path_positions: LongTensor,
                path_weight: Tensor) -> Tuple[Tensor, Tensor]:

        batch_size = x.size(0)
        x = self.path_proj(self.norm(x))
        x = x[:, path_list, :].contiguous() # B * P * h
        edge_emb = self.path_edge_emb(path_edge_type).unsqueeze(0)
        positional_emb = self.path_positional_emb(path_positions).unsqueeze(0)
        x = x + edge_emb + positional_emb
        s = self.aggregate_proj(x).transpose(1, 2).unsqueeze(-1)  # B * r * P * 1

        s = scatter_softmax(s, path_index, dim=-2)  # B * r * P * 1
        h = (x.unsqueeze(1) * s)  # B * r * P * emb
        h = scatter(h, path_index, dim=-2, reduce="sum").permute(0, 2, 1, 3).contiguous()  # B * num_path * r * emb
        h = h.view(batch_size, -1, self.hidden_size)  # B * num_path * H
        h = h * path_weight  # B * num_path * H
        h = torch.mean(h, dim=-2)  # B * H
        return h


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class CentralityEncoding(nn.Module):
    """Compute the centrality encoding for each node in the graph.
    Args:
        num_in_degree (int): number of different in degree.
        num_out_degree (int): number of different out degree.
        hidden_size (int): hidden size of the model.
        N (int): number of layer in the model.
        padding_idx (Optional, int): padding index.
    """

    def __init__(self,
                 num_in_degree: int,
                 num_out_degree: int,
                 hidden_size: int,
                 N: int,
                 padding_idx: Optional[float] = 0):
        super(CentralityEncoding, self).__init__()
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_size, padding_idx=padding_idx)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_size, padding_idx=padding_idx)

        self.apply(lambda module: init_params(module, n_layers=N))

    def forward(self, in_deg: LongTensor, out_deg: LongTensor) -> Tensor:
        ce = self.in_degree_encoder(in_deg) + self.out_degree_encoder(out_deg)
        return ce.squeeze()


class GraphEncodingBias(nn.Module):
    """Compute graph encoding based on shortest path length
    Args:
        num_nodes (int): Number of nodes in the dataset.
        num_edges (int): Number of edge types.
        num_head (int): Number of head in the model.
        N (int): Number of layer in the model.
        padding_idx (Optional, int): Padding index.
    """

    def __init__(self,
                 num_nodes: int,
                 num_edges: int,
                 num_head: int,
                 N: int,
                 padding_idx=0):
        super(GraphEncodingBias, self).__init__()
        self.graph_encoder = nn.Embedding(num_nodes, num_head, padding_idx=None)
        self.edge_encoder = nn.Embedding(num_edges + 1, num_head, padding_idx=padding_idx)
        self.apply(lambda module: init_params(module, n_layers=N))

    def forward(self, node_index: LongTensor, edge_types: LongTensor) -> Tensor:
        # B * N * N
        batch_size = node_index.size(0)
        num_nodes = node_index.size(1)
        node_index = node_index.view(batch_size, -1)  # B * N
        graph_bias = self.graph_encoder(node_index)  # B * N * head
        head = graph_bias.size(-1)
        graph_bias = torch.mul(graph_bias.unsqueeze(-2), graph_bias.unsqueeze(1)).view(batch_size, -1, head)
        edge_types = edge_types.view(batch_size, -1)
        edge_bias = self.edge_encoder(edge_types)
        bias = graph_bias + edge_bias
        bias = bias.view(batch_size, num_nodes, num_nodes, -1).permute(0, 3, 1, 2)
        return bias
