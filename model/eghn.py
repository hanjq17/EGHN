from torch import nn
import torch
import torch.nn.functional as F
from torch_sparse import spmm
from model.basic import EGNN, EquivariantScalarNet, BaseMLP, aggregate, EGMN


class EquivariantEdgeScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, activation, n_scalar_input=0, norm=True, flat=False):
        """
        The universal O(n) equivariant network using scalars.
        :param n_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(EquivariantEdgeScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input * n_vector_input,
                                      self.activation, flat=flat)

    def forward(self, vectors_i, vectors_j, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A vector that is equivariant to the O(n) transformations of input vectors with shape [N, 3]
        """
        Z_i, Z_j = vectors_i, vectors_j  # [N, 3, K]
        K = Z_i.shape[-1]
        Z_j_T = Z_j.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_j_T, Z_i)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.in_scalar_net(scalar)  # [N, H]
        vec_scalar = self.out_vector_net(scalar)  # [N, KK]
        vec_scalar = vec_scalar.reshape(-1, Z_j.shape[-1], Z_i.shape[-1])  # [N, K, K]
        vector = torch.einsum('bij,bjk->bik', Z_j, vec_scalar)  # [N, 3, K]
        return vector, scalar


class PoolingLayer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, n_vector_input, activation=nn.SiLU(), flat=False):
        super(PoolingLayer, self).__init__()
        self.edge_message_net = EquivariantEdgeScalarNet(n_vector_input=n_vector_input, hidden_dim=hidden_nf,
                                                         activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                         norm=True, flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)

    def forward(self, vectors, h, edge_index, edge_fea):
        """
        :param vectors: the node vectors with shape: [BN, 3, V] where V is the number of vectors
        :param h: the scalar node feature with shape: [BN, K]
        :param edge_index: the edge index with shape [2, BM]
        :param edge_fea: the edge feature with shape: [BM, T]
        :return: the updated node vectors [BN, 3, V] and node scalar feature [BN, K]
        """
        row, col = edge_index
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        vectors_i, vectors_j = vectors[row], vectors[col]  # [BM, 3, V]
        vectors_out, message = self.edge_message_net(vectors_i=vectors_i, vectors_j=vectors_j, scalars=hij)  # [BM, 3, V]
        DIM, V = vectors_out.shape[-2], vectors_out.shape[-1]
        vectors_out = vectors_out.reshape(-1, DIM * V)  # [BM, 3V]
        vectors_out = aggregate(message=vectors_out, row_index=row, n_node=h.shape[0], aggr='mean')  # [BN, 3V]
        vectors_out = vectors_out.reshape(-1, DIM, V)  # [BN, 3, V]
        vectors_out = vectors + vectors_out  # [BN, 3, V]
        tot_message = aggregate(message=message, row_index=row, n_node=h.shape[0], aggr='sum')  # [BN, K]
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, K+K]
        h = self.node_net(node_message) + h  # [BN, K]
        return vectors_out, h


class PoolingNet(nn.Module):
    def __init__(self, n_layers, in_edge_nf, n_vector_input,
                 hidden_nf, output_nf, activation=nn.SiLU(), device='cpu', flat=False):
        super(PoolingNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            layer = PoolingLayer(in_edge_nf, hidden_nf, n_vector_input=n_vector_input, activation=activation, flat=flat)
            self.layers.append(layer)
        self.pooling = nn.Sequential(
            nn.Linear(hidden_nf, 8 * hidden_nf),
            nn.Tanh(),
            nn.Linear(8 * hidden_nf, output_nf)
        )
        self.to(device)

    def forward(self, vectors, h, edge_index, edge_fea):
        if type(vectors) == list:
            vectors = torch.stack(vectors, dim=-1)  # [BN, 3, V]
        for i in range(self.n_layers):
            vectors, h = self.layers[i](vectors, h, edge_index, edge_fea)
        pooling = self.pooling(h)
        return pooling  # [BN, P]


class EGHN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, n_cluster, layer_per_block=3, layer_pooling=3, layer_decoder=1,
                 flat=False, activation=nn.SiLU(), device='cpu', norm=False):
        super(EGHN, self).__init__()
        node_hidden_dim = hidden_nf
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.current_pooling_plan = None
        self.n_cluster = n_cluster  # 4 for simulation and 5 for mocap
        self.n_layer_per_block = layer_per_block
        self.n_layer_pooling = layer_pooling
        self.n_layer_decoder = layer_decoder
        self.flat = flat
        # low-level force net
        self.low_force_net = EGNN(n_layers=self.n_layer_per_block,
                                  in_node_nf=hidden_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf,
                                  activation=activation, device=device, with_v=True, flat=flat, norm=norm)
        self.low_pooling = PoolingNet(n_vector_input=3, hidden_nf=hidden_nf, output_nf=self.n_cluster,
                                      activation=activation, in_edge_nf=in_edge_nf, n_layers=self.n_layer_pooling, flat=flat)
        self.high_force_net = EGNN(n_layers=self.n_layer_per_block,
                                   in_node_nf=hidden_nf, in_edge_nf=1, hidden_nf=hidden_nf,
                                   activation=activation, device=device, with_v=True, flat=flat)
        if self.n_layer_decoder == 1:
            self.kinematics_net = EquivariantScalarNet(n_vector_input=4,
                                                       hidden_dim=hidden_nf,
                                                       activation=activation,
                                                       n_scalar_input=node_hidden_dim + node_hidden_dim,
                                                       norm=True,
                                                       flat=flat)
        else:
            self.kinematics_net = EGMN(n_vector_input=4, hidden_dim=hidden_nf, activation=activation,
                                       n_scalar_input=node_hidden_dim + node_hidden_dim, norm=True, flat=flat,
                                       n_layers=self.n_layer_decoder)

        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, local_edge_index, local_edge_fea, n_node,  v=None, node_mask=None, node_nums=None):
        """
        :param x: input positions [B * N, 3]
        :param h: input node feature [B * N, R]
        :param edge_index: edge index of the graph [2, B * M]
        :param edge_fea: input edge feature [B* M, T]
        :param local_edge_index: the edges used in pooling network [B * M']
        :param local_edge_fea: the feature of local edges [B * M', T]
        :param n_node: number of nodes per graph [1, ]
        :param v: input velocities [B * N, 3] (Optional)
        :param node_mask: the node mask when number of nodes are different in graphs [B * N, ] (Optional)
        :param node_nums: the real number of nodes in each graph
        :return:
        """
        h = self.embedding(h)  # [R, K]
        row, col = edge_index

        ''' low level force '''
        new_x, new_v, h = self.low_force_net(x, h, edge_index, edge_fea, v=v)  # [BN, 3]
        nf = new_x - x  # [BN, 3]

        ''' pooling network '''
        if node_nums is None:
            x_mean = torch.mean(x.reshape(-1, n_node, x.shape[-1]), dim=1, keepdim=True).expand(-1, n_node, -1).reshape(
                -1, x.shape[-1])
        else:
            pooled_mean = (torch.sum(x.reshape(-1, n_node, x.shape[-1]), dim=1).T/node_nums).T.unsqueeze(dim=1) #[B,1,3]
            x_mean = pooled_mean.expand(-1, n_node, -1).reshape(-1, x.shape[-1])

        pooling_fea = self.low_pooling(vectors=[x - x_mean, nf, v], h=h,
                                       edge_index=local_edge_index, edge_fea=local_edge_fea)  # [BN, P]

        hard_pooling = pooling_fea.argmax(dim=-1)
        hard_pooling = F.one_hot(hard_pooling, num_classes=self.n_cluster).float()
        pooling = F.softmax(pooling_fea, dim=1)
        self.current_pooling_plan = hard_pooling  # record the pooling plan

        ''' derive high-level information (be careful with graph mini-batch) '''
        s = pooling.reshape(-1, n_node, pooling.shape[-1])  # [B, N, P]

        sT = s.transpose(-2, -1)  # [B, P, N]
        p_index = torch.ones_like(nf)[..., 0]  # [BN, ]
        if node_mask is not None:
            p_index = p_index * node_mask
        p_index = p_index.reshape(-1, n_node, 1)  # [B, N, 1]
        count = torch.einsum('bij,bjk->bik', sT, p_index).clamp_min(1e-5)  # [B, P, 1]
        _x, _h, _nf = x.reshape(-1, n_node, x.shape[-1]), h.reshape(-1, n_node, h.shape[-1]), nf.reshape(-1, n_node, nf.shape[-1])
        _v = v.reshape(-1, n_node, v.shape[-1])
        # [B, N, 3], [B, N, K], [B, N, 3]
        X, H, NF = torch.einsum('bij,bjk->bik', sT, _x), torch.einsum('bij,bjk->bik', sT, _h), torch.einsum('bij,bjk->bik', sT, _nf)
        V = torch.einsum('bij,bjk->bik', sT, _v)
        X, H, NF, V = X / count, H / count, NF / count, V / count  # [B, P, 3], [B, P, K], [B, P, 3]
        X, H, NF = X.reshape(-1, X.shape[-1]), H.reshape(-1, H.shape[-1]), NF.reshape(-1, NF.shape[-1])  # [BP, 3]
        V = V.reshape(-1, V.shape[-1])
        a = spmm(torch.stack((local_edge_index[0], local_edge_index[1]), dim=0),
                 torch.ones_like(local_edge_index[0]), x.shape[0], x.shape[0], pooling)  # [BN, P]
        a = a.reshape(-1, n_node, a.shape[-1])  # [B, N, P]
        A = torch.einsum('bij,bjk->bik', sT, a)  # [B, P, P]
        self.cut_loss = self.get_cut_loss(A)
        aa = spmm(torch.stack((row, col), dim=0), torch.ones_like(row), x.shape[0], x.shape[0], pooling)  # [BN, P]
        aa = aa.reshape(-1, n_node, aa.shape[-1])  # [B, N, P]
        AA = torch.einsum('bij,bjk->bik', sT, aa)  # [B, P, P]

        # construct high-level edges
        h_row, h_col, h_edge_fea, h_edge_mask = self.construct_edges(AA, AA.shape[-1])  # [BPP]
        ''' high-level message passing '''
        h_new_x, h_new_v, h_new_h = self.high_force_net(X, H, (h_row, h_col), h_edge_fea.unsqueeze(-1), v=V)
        h_nf = h_new_x - X

        ''' high-level kinematics update '''
        _X = X + h_nf  # [BP, 3]
        _V = h_new_v  # [BP, 3]
        _H = h_new_h  # [BP, K]

        ''' low-level kinematics update '''
        l_nf = h_nf.reshape(-1, AA.shape[1], h_nf.shape[-1])  # [B, P, 3]
        l_nf = torch.einsum('bij,bjk->bik', s, l_nf).reshape(-1, l_nf.shape[-1])  # [BN, 3]
        l_X = X.reshape(-1, AA.shape[1], X.shape[-1])  # [B, P, 3]
        l_X = torch.einsum('bij,bjk->bik', s, l_X).reshape(-1, l_X.shape[-1])  # [BN, 3]
        l_V = V.reshape(-1, AA.shape[1], V.shape[-1])  # [B, P, 3]
        l_V = torch.einsum('bij,bjk->bik', s, l_V).reshape(-1, l_V.shape[-1])  # [BN, 3]
        l_H = _H.reshape(-1, AA.shape[1], _H.shape[-1])  # [B, P, K]
        l_H = torch.einsum('bij,bjk->bik', s, l_H).reshape(-1, l_H.shape[-1])  # [BN, K]
        l_kinematics, h_out = self.kinematics_net(vectors=[l_nf, x - l_X, v - l_V, nf],
                                                  scalars=torch.cat((h, l_H), dim=-1))  # [BN, 3]
        _l_X = _X.reshape(-1, AA.shape[1], _X.shape[-1])  # [B, P, 3]
        _l_X = torch.einsum('bij,bjk->bik', s, _l_X).reshape(-1, _l_X.shape[-1])  # [BN, 3]
        x_out = _l_X + l_kinematics  # [BN, 3]

        return (x_out, v, h_out) if v is not None else (x_out, h_out)

    def inspect_pooling_plan(self):
        plan = self.current_pooling_plan  # [BN, P]
        if plan is None:
            print('No pooling plan!')
            return
        dist = torch.sum(plan, dim=0)  # [P,]
        # print(dist)
        dist = F.normalize(dist, p=1, dim=0)  # [P,]
        print('Pooling plan:', dist.detach().cpu().numpy())
        return

    def get_cut_loss(self, A):
        A = F.normalize(A, p=2, dim=2)
        return torch.norm(A - torch.eye(A.shape[-1]).to(A.device), p="fro", dim=[1, 2]).mean()

    @staticmethod
    def construct_edges(A, n_node):
        h_edge_fea = A.reshape(-1)  # [BPP]
        h_row = torch.arange(A.shape[1]).unsqueeze(-1).expand(-1, A.shape[1]).reshape(-1).to(A.device)
        h_col = torch.arange(A.shape[1]).unsqueeze(0).expand(A.shape[1], -1).reshape(-1).to(A.device)
        h_row = h_row.unsqueeze(0).expand(A.shape[0], -1)
        h_col = h_col.unsqueeze(0).expand(A.shape[0], -1)
        offset = (torch.arange(A.shape[0]) * n_node).unsqueeze(-1).to(A.device)
        h_row, h_col = (h_row + offset).reshape(-1), (h_col + offset).reshape(-1)  # [BPP]
        h_edge_mask = torch.ones_like(h_row)  # [BPP]
        h_edge_mask[torch.arange(A.shape[1]) * (A.shape[1] + 1)] = 0
        return h_row, h_col, h_edge_fea, h_edge_mask


