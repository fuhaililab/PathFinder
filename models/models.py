"""
Implementation of PathFormer Model in PyTorch
Jiarui Feng
"""

from models.layers import *
from copy import deepcopy as c

EPS = 1e-15


class PathFinder(nn.Module):
    r"""PathFinder model class.
    Args:
        input_size (int): Input gene feature size of the model.
        hidden_size (int): Hidden size of the model.
        r (int): Inner projection size for path embedding.
        N (int): Number of layer in the model.
        head (int): Number of head.
        num_in_degree (int): Maximum number of in-degree in the graph.
        num_out_degree (int): Maximum number of out-degree in the graph.
        num_nodes (int): Number of genes in the dataset.
        num_paths (int): Number of paths in the dataset.
        num_edges (int): Number of edge type in the dataset.
        max_path_len (int): Maximum length of the input path.
        gamma (float): Gamma value in the high-order softmax.
        JK (str): Jumping knowledge method.
        drop_prob (float): Dropout probability.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 r: int,
                 N: int,
                 head: int,
                 num_in_degree: int,
                 num_out_degree: int,
                 num_nodes: int,
                 num_paths: int,
                 num_edges: int,
                 max_path_len: int,
                 gamma: float,
                 JK: str,
                 drop_prob: float):
        super(PathFinder, self).__init__()
        self.gamma = gamma
        self.N = N
        self.JK = JK
        self.initial_proj = nn.Linear(input_size, hidden_size)
        self.ce = CentralityEncoding(num_in_degree=num_in_degree,
                                     num_out_degree=num_out_degree,
                                     hidden_size=hidden_size,
                                     N=N)
        self.geb = GraphEncodingBias(num_nodes=num_nodes,
                                     num_edges=num_edges,
                                     num_head=head,
                                     N=N)
        attn = MultiHeadedAttention(hidden_size=hidden_size,
                                    h=head,
                                    drop_prob=drop_prob)
        FFN = PointwiseFeedForwardNetwork(hidden_size=hidden_size,
                                          drop_prob=drop_prob)
        path_embedding = PathEmbedding(hidden_size=hidden_size,
                                       r=r,
                                       max_path_len=max_path_len,
                                       num_edges=num_edges,
                                       drop_prob=drop_prob)
        encoder_layer = EncoderLayer(hidden_size=hidden_size,
                                     self_attn=c(attn),
                                     feed_forward=c(FFN),
                                     path_embedding_layer=c(path_embedding),
                                     drop_prob=drop_prob)
        self.encoder_block = EncoderBlock(layer=encoder_layer,
                                          N=N)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * num_paths))
        self.path_weight = nn.Parameter(torch.rand([1, num_paths, 1]) * std)

    def forward(self,
                x: Tensor,
                mask: Tensor,
                in_deg: LongTensor,
                out_deg: LongTensor,
                edge_types: LongTensor,
                node_index: LongTensor,
                path_list: LongTensor,
                path_index: LongTensor,
                path_edge_type: LongTensor,
                path_positions: LongTensor) -> Tensor:
        """
        x(B * N * H)
        mask(B * N)
        in_deg (B * N * 1)
        out_deg: (B * N * 1)
        edge_types: (B * N * N)
        node_index: (B * N)
        path_list: (P)
        path_index: (P)
        """
        x = self.initial_proj(x)
        # centrality encoding
        x = x + self.ce(in_deg, out_deg)
        # graph encoding bias
        bias = self.geb(node_index, edge_types)
        path_weight = self.return_path_weight()
        # pathFormer forwarding
        x, h_list = self.encoder_block(x, bias, mask, path_list, path_index, path_edge_type, path_positions, path_weight)

        # JK connection
        if self.JK == "concat":
            path_emb = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            path_emb = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            path_emb = F.max_pool1d(torch.cat(h_list, dim=-1), kernel_size=self.N).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            path_emb = torch.sum(torch.cat(h_list, dim=0), dim=0)
        return x, path_emb

    def compute_reg_loss(self, prior_path_weight: Tensor) -> Tensor:
        """
        #size loss


        #input loss
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        input_loss = ent.mean()
        """
        path_weight = self.return_path_weight().squeeze()
        size_loss = F.binary_cross_entropy(path_weight,
                                           torch.zeros_like(path_weight, device=self.path_weight.device))
        piror_loss = F.binary_cross_entropy(path_weight, prior_path_weight.squeeze())
        #return F.kl_div(torch.log(self.high_order_softmax().squeeze()), prior_path_weight.squeeze(), reduction="sum")
        return size_loss + piror_loss

    def return_path_weight(self) -> Tensor:
        return torch.sigmoid(self.path_weight)

    # def high_order_softmax(self) -> Tensor:
    #     s = torch.softmax(self.path_weight, dim=-2)
    #     s = s ** self.gamma
    #     s = s / torch.sum(s)
    #     return s

    # def binary_concrete_distribution(self) -> Tensor:
    #     path_weight = self.path_weight
    #     u = torch.rand(path_weight.size(), device=path_weight.device)
    #     m = torch.sigmoid((torch.log(u) - torch.log(1 - u) + path_weight) / self.tem)  #
    #     return m


class SimpleClassifier(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 output_size: int,
                 drop_prob: Optional[float] = 0.1):
        super(SimpleClassifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(drop_prob),
                                 nn.Linear(hidden_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
