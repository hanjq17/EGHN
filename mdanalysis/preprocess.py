import os
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix

from MDAnalysisData import datasets
import MDAnalysis
from MDAnalysis.analysis import distances


def compute_ele(ts):
    edge = coo_matrix(distances.contact_matrix(ts.positions, cutoff=cut_off, returntype="sparse"))
    edge.setdiag(False)
    edge.eliminate_zeros()
    edge_global = [torch.tensor(edge.row, dtype=torch.long), torch.tensor(edge.col, dtype=torch.long)]
    global_edge_attr = torch.norm(torch.tensor(ts.positions[edge.row, :] - ts.positions[edge.col, :]), p=2, dim=1)
    return edge_global, global_edge_attr

delta_frame = 50
tmp_dir = 'YOUR_MD_DATA_DIR/dataset/mdanalysis/'
cut_off = 10
train_valid_test_ratio = [0.6, 0.2, 0.2]

adk = datasets.fetch_adk_equilibrium(data_home=tmp_dir)
data = MDAnalysis.Universe(adk.topology, adk.trajectory)

train_valid_test = [int(train_valid_test_ratio[0] * len(data.trajectory)),
                    int(sum(train_valid_test_ratio[:2]) * len(data.trajectory))]


charges = torch.tensor(data.atoms.charges)
edges = [torch.tensor(data.bonds.indices[:, 0], dtype=torch.long),
         torch.tensor(data.bonds.indices[:, 1], dtype=torch.long)]

edge_attr = torch.tensor([bond.length() for bond in data.bonds])

loc = []
vel = []

for i in tqdm(range(len(data.trajectory) - 1)):
    loc.append(torch.tensor(data.trajectory[i].positions))
    vel.append(torch.tensor(data.trajectory[i + 1].positions - data.trajectory[i].positions))

torch.save((edges, edge_attr, charges, len(data.trajectory) - 1), os.path.join(tmp_dir, 'adk_processed', 'adk.pkl'))

edges_global, edges_global_attr = zip(*Parallel(n_jobs=-1)(delayed(compute_ele)(_) for _ in tqdm(data.trajectory)))
edges_global = edges_global[:-1]
edges_global_attr = edges_global_attr[:-1]


for i in tqdm(range(len(loc))):
    try:
        torch.save((loc[i], vel[i], edges_global[i], edges_global_attr[i]),
                   os.path.join(tmp_dir, 'adk_processed', f'adk_{i}.pkl'))
    except RuntimeError:
         print(i)

