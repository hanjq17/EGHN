import argparse
import torch
import torch.utils.data
from motion.dataset import MotionDataset
from model.eghn import EGHN
import os
from torch import nn, optim
import json

import random
import numpy as np

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='exp_results', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='hier', metavar='N')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--delta_frame', type=int, default=50,
                    help='Number of frames delta.')
parser.add_argument('--data_dir', type=str, default='spatial_graph/md17',
                    help='Data directory.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--config_by_file", default=False, action="store_true", )

parser.add_argument('--lambda_link', type=float, default=1,
                    help='The weight of the linkage loss.')
parser.add_argument('--n_cluster', type=int, default=3,
                    help='The number of clusters.')
parser.add_argument('--flat', action='store_true', default=False,
                    help='flat MLP')
parser.add_argument('--interaction_layer', type=int, default=3,
                    help='The number of interaction layers per block.')
parser.add_argument('--pooling_layer', type=int, default=3,
                    help='The number of pooling layers in EGPN.')
parser.add_argument('--decoder_layer', type=int, default=1,
                    help='The number of decoder layers.')

parser.add_argument('--case', type=str, default='walk',
                    help='The case, walk or run.')

time_exp_dic = {'time': 0, 'counter': 0}

args = parser.parse_args()
# Place the checkpoint file here
ckpt_file = args.outf + '/' + args.exp_name + '/' + 'saved_model.pth'

if args.config_by_file:
    job_param_path = './job_param.json'
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        args.exp_name = hyper_params["exp_name"]
        args.batch_size = hyper_params["batch_size"]
        args.epochs = hyper_params["epochs"]
        args.seed = hyper_params["seed"]
        args.lr = hyper_params["lr"]
        args.nf = hyper_params["nf"]
        args.model = hyper_params["model"]
        args.n_layers = hyper_params["n_layers"]
        args.max_training_samples = hyper_params["max_training_samples"]
        # Do not necessary in practice.
        args.data_dir = hyper_params["data_dir"]
        args.weight_decay = hyper_params["weight_decay"]

        args.dropout = hyper_params["dropout"]
        args.lambda_link = hyper_params["lambda_link"]
        args.n_cluster = hyper_params["n_cluster"]
        args.flat = hyper_params["flat"]
        args.interaction_layer = hyper_params["interaction_layer"]
        args.pooling_layer = hyper_params["pooling_layer"]
        args.decoder_layer = hyper_params["decoder_layer"]

        args.case = hyper_params["case"]

args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)


def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_train = MotionDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame, case=args.case)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                               num_workers=8)

    dataset_val = MotionDataset(partition='val', max_samples=600, data_dir=args.data_dir,
                                delta_frame=args.delta_frame, case=args.case)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=8)

    dataset_test = MotionDataset(partition='test', max_samples=600, data_dir=args.data_dir,
                                 delta_frame=args.delta_frame, case=args.case)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    if args.model == 'hier':
        model = EGHN(in_node_nf=2, in_edge_nf=2, hidden_nf=args.nf, device=device,
                     n_cluster=args.n_cluster, flat=args.flat, layer_per_block=args.interaction_layer,
                     layer_pooling=args.pooling_layer, activation=nn.SiLU(),
                     layer_decoder=args.decoder_layer)
        model.load_state_dict(torch.load(ckpt_file))
        print('loaded from ', ckpt_file)
    else:
        raise Exception("Wrong model specified")

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.eval()
    save_name = args.outf + '/' + args.exp_name + '/' + 'eval_train.pkl'
    train_loss = train(model, optimizer, 0, loader_train, backprop=False, save_name=save_name)
    save_name = args.outf + '/' + args.exp_name + '/' + 'eval_test.pkl'
    test_loss = train(model, optimizer, 0, loader_test, backprop=False, save_name=save_name)
    exit(0)

    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True, save_name=None):
    all_loc, all_loc_pred, all_loc_end = None, None, None
    all_pooling_plan = None
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        # data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        loc, vel, edges, edge_attr, local_edges, local_edge_fea, Z, loc_end, vel_end = data
        # convert into graph minibatch
        loc = loc.view(-1, loc.size(2))
        vel = vel.view(-1, vel.size(2))
        offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        edges = torch.cat(list(edges + offset), dim=-1)  # [2, BM]
        edge_attr = torch.cat(list(edge_attr), dim=0)  # [BM, ]
        local_edge_index = torch.cat(list(local_edges + offset), dim=-1)  # [2, BM]
        local_edge_fea = torch.cat(list(local_edge_fea), dim=0)  # [BM, ]
        # local_edge_mask = torch.cat(list(local_edge_mask), dim=0)  # [BM, ]
        Z = Z.view(-1, Z.size(2))
        loc_end = loc_end.view(-1, loc_end.size(2))
        vel_end = vel_end.view(-1, vel_end.size(2))

        if all_loc is None:
            all_loc = loc
        else:
            all_loc = torch.cat((all_loc, loc), dim=0)

        optimizer.zero_grad()

        if args.model == 'hier':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_dist1 = torch.sum((loc[local_edge_index[0]] - loc[local_edge_index[1]])**2, 1).unsqueeze(1)
            local_edge_fea = torch.cat([local_edge_fea, loc_dist1], 1).detach()  # concatenate all edge properties
            loc_pred, vel_pred, _ = model(loc, nodes, edges, edge_attr, local_edge_index, local_edge_fea,
                                          n_node=n_nodes, v=vel, node_mask=None, node_nums=None)
        else:
            raise Exception("Wrong model")

        if all_loc_pred is None:
            all_loc_pred = loc_pred
        else:
            all_loc_pred = torch.cat((all_loc_pred, loc_pred), dim=0)

        if all_loc_end is None:
            all_loc_end = loc_end
        else:
            all_loc_end = torch.cat((all_loc_end, loc_end), dim=0)

        cur_pooling_plan = model.current_pooling_plan
        if all_pooling_plan is None:
            all_pooling_plan = cur_pooling_plan
        else:
            all_pooling_plan = torch.cat((all_pooling_plan, cur_pooling_plan), dim=0)

        loss = loss_mse(loc_pred, loc_end)

        if backprop:
            loss.backward()
            optimizer.step()
            pass
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
    import pickle as pkl
    with open(save_name, 'wb') as f:
        pkl.dump((all_loc.detach().cpu().numpy(),
                  all_loc_end.detach().cpu().numpy(),
                  all_loc_pred.detach().cpu().numpy(),
                  all_pooling_plan.detach().cpu().numpy()
                  ), f)
    print('Saved to ', save_name)

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f'
          % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
    print("best_train = %.6f, best_val = %.6f, best_test = %.6f, best_epoch = %d" % (best_train_loss, best_val_loss, best_test_loss, best_epoch))





