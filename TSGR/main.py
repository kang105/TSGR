#! /usr/bin/env python

import torch
from skimage.metrics import structural_similarity as ssim
import scipy.io as scio
import time
from torch.utils.tensorboard import SummaryWriter
from model_TSGR import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util import *

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda")
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
np.random.seed(123)

# Parameters
# ==================================================
NUM = 360        #number of graph nodes

parser = ArgumentParser("TSGR", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default=["path of graph node features  .txt",
                                          "path of graph adj  .txt"], help="Path of the dataset.")
parser.add_argument("--num_list", default="  .txt", help="The individual number file of all hcp data used")
parser.add_argument("--num_node", default=NUM, help="number of node")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=10, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=200, type=int, help="Number of training epochs")

parser.add_argument('--sampled_num', default=NUM, type=int, help="The size of the node embedding")
parser.add_argument("--dropout", default=0.5, type=float, help="")

parser.add_argument("--num_hidden_layers", default=1, type=int, help="Number of GCN layers")
parser.add_argument("--num_timesteps", default=1, type=int, help="Number of self-attention layers within each UGformer layer")
parser.add_argument("--ff_hidden_size", default=NUM * 2, type=int, help="The hidden size for the feedforward layer")

parser.add_argument('--mask_ratio', type=float, default=0.5, help="")
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--encoder_self', type=int, default=1, help="Number of UGformer layers of encoder")
parser.add_argument('--decoder_self', type=int, default=1, help="Number of UGformer layers of decoder")
parser.add_argument('--l_gamma', type=float, default=0.5)
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")
graphs = load_data(args.dataset, args.num_list, 0)
print(len(graphs))
feature_dim_size = graphs[0].node_features.shape[1]
print(feature_dim_size)
print("Loading data... finished!")


model = TSGR(feature_dim_size=feature_dim_size, feature_embedding_size=args.sampled_num, ff_hidden_size=args.ff_hidden_size,
                        dropout=args.dropout, num_self_att_layers=args.num_timesteps,
                        num_GNN_layers_1=args.num_hidden_layers, num_GNN_layers_2=args.num_hidden_layers, nhead=args.nhead, e_s=args.encoder_self, d_s=args.decoder_self).to(device)

def loss_fun(x_original, x_recon, graph_pool):
    loss_value = torch.tensor(0)
    for i in range(len(graph_pool)):
        one_graph_idx = graph_pool[i]._indices()[0]
        start_idx = one_graph_idx[0]
        end_idx = one_graph_idx[len(one_graph_idx) - 1] + 1
        one_graph = x_original[start_idx: end_idx]
        one_graph_re = x_recon[start_idx: end_idx]
        loss_value = loss_value + torch.sum(torch.square(one_graph - one_graph_re)) / one_graph.size(0)
    return loss_value

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=args.l_gamma)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    #shuffle
    indices = np.arange(0, len(graphs))
    np.random.shuffle(indices)
    batch_score_list = []
    batch_out_list = []
    torch.autograd.set_detect_anomaly(True)

    for start in range(0, len(graphs), args.batch_size):
        end = start + args.batch_size
        selected_idx = indices[start:end]
        batch_graph = [graphs[idx] for idx in selected_idx]
        input_x, graph_pool, X_concat, adj = get_batch_data(batch_graph)
        optimizer.zero_grad()
        reconstruct, batch_score = model(X_concat, graph_pool, args.mask_ratio, adj, NUM)
        loss = loss_fun(X_concat, reconstruct, graph_pool)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        batch_score_list.append(batch_score)
        batch_out_list.append(reconstruct)

    all_score = torch.cat([score for score in batch_score_list], dim=0)
    all_out = torch.cat([out for out in batch_out_list], dim=0)
    return total_loss, all_score, all_out, indices



"""main process"""
import os

out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_Brain_UnSup_Generalize", args.model_name))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints_%s' % time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


write = SummaryWriter(comment=' ', log_dir="")
cost_loss = []
epoch_index_list = []
epoch_ssim_list =[]

for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss, train_score, out, index = train()

    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} |'.format(
                epoch, (time.time() - epoch_start_time), train_loss))

    write.add_scalar('train_loss', train_loss, epoch)

    # To calculate SSIM between the reconstructed and the original feature matrices
    all_out = out.reshape(len(graphs), NUM, NUM).data.cpu().numpy()
    index = np.reshape(index, (1, len(graphs)))
    all_out = all_out[np.argsort(index[0])]
    all_ssim = [ssim(all_out[s], graphs[s].node_features) for s in range(len(graphs))]
    all_ssim = np.reshape(np.array(all_ssim), (1, len(graphs)))

    epoch_ssim_list.append(all_ssim)
    epoch_index_list.append(index)
    cost_loss.append(train_loss)

#     path_name = " "
#     if os.path.exists(path_name) == False:
#         os.makedirs(path_name)
#     save_name = path_name + ('\epoch_%d.mat' % epoch)
#     scio.savemat(save_name,
#                  {'train_score': train_score.detach().cpu().numpy()})
#
# epoch_index = np.concatenate(epoch_index_list, axis=0)
# epoch_ssim = np.concatenate(epoch_ssim_list, axis=0)
#
# scio.savemat(path_name + '\index.mat', {'epoch_index': epoch_index})
# scio.savemat(path_name + '\ssim.mat', {'ssim': epoch_ssim})
# scio.savemat(path_name + '\loss.mat', {'loss': np.array(cost_loss)})

write.close()
