import numpy as np
import torch
import random
import pickle as pkl
from functools import reduce


class SimulationDataset():
    """
    NBodyDataset

    """
    def __init__(self, partition='train', max_samples=1e8,
                 data_dir='', n_complex=5, average_complex_size=3, system_types=5):
        self.partition = partition
        self.data_dir = data_dir
        self.n_complex = n_complex
        self.average_complex_size = average_complex_size
        self.system_types = system_types

        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition

        self.suffix += '_charged{:d}_{:d}_{:d}'.format(n_complex, average_complex_size, system_types)
        # self.suffix += '_charged0_0_0_3'

        self.max_samples = int(max_samples)
        self.loc, self.vel, self.charges, self.edges, self.cfg = self.load()
        # self.data, self.edges, self.cfg = self.load()

    def load(self):
        # loc = np.load(self.data_dir + '/' + 'loc_' + self.suffix + '.npy')  # [N_SAMPLE, N_FRAME, N_NODE, 3]
        # vel = np.load(self.data_dir + '/' + 'vel_' + self.suffix + '.npy')
        # charges = np.load(self.data_dir + '/' + 'charges_' + self.suffix + '.npy')
        # edges = np.load(self.data_dir + '/' + 'edges_' + self.suffix + '.npy')

        with open(self.data_dir + '/' + 'loc_' + self.suffix + '.pkl', 'rb') as f:  # [N_SAMPLE, N_FRAME, N_NODE, 3]
            loc = pkl.load(f)
        with open(self.data_dir + '/' + 'vel_' + self.suffix + '.pkl', 'rb') as f:
            vel = pkl.load(f)
        with open(self.data_dir + '/' + 'charges_' + self.suffix + '.pkl', 'rb') as f:
            charges = pkl.load(f)
        with open(self.data_dir + '/' + 'edges_' + self.suffix + '.pkl', 'rb') as f:
            edges = pkl.load(f)
        with open(self.data_dir + '/' + 'cfg_' + self.suffix + '.pkl', 'rb') as f:
            cfg = pkl.load(f)

        loc = loc[:self.max_samples]
        vel = vel[:self.max_samples]
        charges = charges[: self.max_samples]
        edges = edges[: self.max_samples]

        return loc, vel, charges, edges, cfg

        # loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        # return (loc, vel, edge_attr, charges), edges, cfg

    # def preprocess(self, loc, vel, edges, charges):
    #     loc, vel = torch.Tensor(loc), torch.Tensor(vel)  # remove transpose this time
    #     n_nodes = loc.size(2)
    #     loc = loc[:self.max_samples, :, :, :]  # limit number of samples
    #     vel = vel[:self.max_samples, :, :, :]  # speed when starting the trajectory
    #     charges = charges[: self.max_samples]
    #     # edges: charge_i * charge_j (edge_attr)
    #     edges = edges[: self.max_samples, ...]  # add here for better consistency
    #     edge_attr = []
    #
    #     # Initialize edges and edge_attributes
    #     rows, cols = [], []
    #     for i in range(n_nodes):
    #         for j in range(n_nodes):
    #             if i != j:  # remove self loop
    #                 edge_attr.append(edges[:, i, j])
    #                 rows.append(i)
    #                 cols.append(j)
    #     edges = [rows, cols]
    #
    #     # swap n_nodes <--> batch_size and add nf dimension
    #     edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)  # [B, N*(N-1), 1]
    #
    #     return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)
    #
    # def set_max_samples(self, max_samples):
    #     self.max_samples = int(max_samples)
    #     self.data, self.edges, self.cfg = self.load()

    def __getitem__(self, i):
        loc, vel, charges, edges, cfg = self.loc[i], self.vel[i], self.charges[i], self.edges[i], self.cfg[i]

        # frame_0, frame_T = 30, 40
        frame_0, frame_T = 10, 25

        edge_attr = []
        n_nodes = loc.shape[1]
        # Initialize edges and edge_attributes for interaction forces
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # remove self loop
                    edge_attr.append(edges[i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).unsqueeze(-1)  # [N*(N-1), 1]

        assert 'Stick' not in cfg and 'Hinge' not in cfg  # Currently, only want to support isolated and complex bodies
        # add edge indicator for Complex
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        if 'Complex' in cfg:
            configs = cfg['Complex']
            for comp in configs:
                # add fully connected graph over the complex body
                for _i in range(len(comp)):
                    for _j in range(len(comp)):
                        if _i != _j:
                            idi, idj = comp[_i], comp[_j]
                            n_node = loc.shape[1]
                            edge_idx = idi * (n_node - 1) + idj
                            if idj > idi:
                                edge_idx -= 1
                            assert edges[0][edge_idx] == idi and edges[1][edge_idx] == idj
                            stick_ind[edge_idx] = 1
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)

        edges = torch.from_numpy(np.array(edges))
        local_edge_mask = edge_attr[..., -1] == 1

        return torch.Tensor(loc[frame_0]), torch.Tensor(vel[frame_0]), edges, edge_attr, local_edge_mask,\
               torch.Tensor(charges), torch.Tensor(loc[frame_T]), torch.Tensor(vel[frame_T])

    def __len__(self):
        return len(self.loc)

    # def get_edges(self, batch_size, n_nodes):
    #     edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
    #     if batch_size == 1:
    #         return edges
    #     elif batch_size > 1:
    #         offset = torch.arange(batch_size) * n_nodes
    #         row = edges[0].unsqueeze(0).repeat(batch_size, 1)
    #         row = row + offset.unsqueeze(-1).expand_as(row)
    #         col = edges[1].unsqueeze(0).repeat(batch_size, 1)
    #         col = col + offset.unsqueeze(-1).expand_as(col)
    #         edges = [row.reshape(-1), col.reshape(-1)]
    #     return edges
    #
    # @staticmethod
    # def get_cfg(batch_size, n_nodes, cfg):
    #     offset = torch.arange(batch_size) * n_nodes
    #     for type in cfg:
    #         index = cfg[type]  # [B, n_type, node_per_type]
    #         cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
    #         if type == 'Isolated':
    #             cfg[type] = cfg[type].squeeze(-1)
    #     return cfg
