import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from gcn_layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TSGR(nn.Module):

    def __init__(self, feature_dim_size, feature_embedding_size,ff_hidden_size, dropout,
                 num_self_att_layers, num_GNN_layers_1, num_GNN_layers_2, nhead, e_s, d_s):
        super(TSGR, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.feature_embedding_size = feature_embedding_size
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers #Each layer consists of a number of self-attention layers
        self.num_GNN_layers_1 = num_GNN_layers_1
        self.num_GNN_layers_2 = num_GNN_layers_2
        #
        self.ugformer_layers = torch.nn.ModuleList()     #这个模型的encoder层的一部分
        self.ugformer_layers_decoder = torch.nn.ModuleList()  #这个模型的decoder层的一部分
        #AutoEncoder
        self.ugformer_layers.append(nn.Linear(self.feature_dim_size,self.feature_embedding_size))
        self.ugformer_layers.append(
            GraphBaseBlock([self.feature_embedding_size, self.feature_embedding_size], 1, dropout=0.5, withloop=False))

        self.pooling_layers = TransformerEncoder(TransformerEncoderLayer(d_model=self.feature_embedding_size, nhead=nhead, dim_feedforward=self.ff_hidden_size, dropout=0.5), e_s)# embed_dim must be divisible by num_heads
        self.linear = nn.Linear(self.feature_embedding_size, 1)

        #Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, self.feature_embedding_size))
        self.depooling_layers = TransformerEncoder(
            TransformerEncoderLayer(d_model=self.feature_embedding_size, nhead=nhead,
                                    dim_feedforward=self.ff_hidden_size, dropout=0.5),
            d_s)  # embed_dim must be divisible by num_heads
        self.ugformer_layers_decoder.append(
            GraphBaseBlock([self.feature_embedding_size, self.feature_embedding_size], 1, dropout=0.5, withloop=False))

        self.ugformer_layers_decoder.append(nn.Linear(self.feature_embedding_size,self.feature_dim_size))
        # Linear function
        self.dropouts = nn.Dropout(dropout)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, X_concat, graph_pool, mask_ratio, adj, NUM):

        X_concat = self.ugformer_layers[0](X_concat)
        X_concat = torch.reshape(X_concat, (len(graph_pool), NUM, X_concat.shape[-1]))
        encoder_graph = X_concat.permute(1, 0, 2)
        self_score = self.pooling_layers(encoder_graph).permute(1, 0, 2)
        self_score = torch.reshape(self_score, (adj.shape[0], self_score.shape[-1]))
        middle_Tr = self.ugformer_layers[1](self_score, adj)
        middle_Tr = torch.reshape(middle_Tr, (len(graph_pool), NUM, middle_Tr.shape[-1]))
        score = self.linear(middle_Tr).squeeze(2)
        softmax = nn.Softmax(dim=1)
        score = softmax(score)
        batch_score = score
        #cat后排序
        del_index = torch.argsort(score)[:, 0: int(NUM * mask_ratio)]
        save_index = torch.argsort(score)[:, int(NUM * mask_ratio):]
        all_index = torch.cat([save_index, del_index], dim=1)
        resort = torch.argsort(all_index)
        save_x = torch.gather(middle_Tr, dim=1, index=save_index.unsqueeze(-1).expand(-1, -1, middle_Tr.shape[-1]))
        add_mask = self.mask_token.repeat(len(graph_pool), del_index.shape[1], 1)
        new_middle_Tr = torch.cat([save_x, add_mask], dim=1)
        new_middle_Tr = torch.gather(new_middle_Tr, dim=1, index=resort.unsqueeze(-1).expand(-1, -1, middle_Tr.shape[-1]))
        output_Tr = new_middle_Tr * score.unsqueeze(-1)
        # decoder

        input_graph = torch.reshape(output_Tr, (len(graph_pool), NUM, X_concat.shape[-1])).permute(1, 0, 2)
        output_Tr = self.depooling_layers(input_graph).permute(1, 0, 2)
        output_Tr = torch.reshape(output_Tr, (len(graph_pool) * NUM, output_Tr.shape[-1]))
        output_Tr = self.ugformer_layers_decoder[0](output_Tr, adj)
        output_Tr = self.ugformer_layers_decoder[-1](output_Tr)

        return output_Tr, batch_score



