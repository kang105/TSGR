import networkx as nx
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class S2VGraph(object):
    def __init__(self, g, node_features):
        '''
            g: a networkx graph
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.g = g
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0
        self.max_neighbor = 0


def load_data(dataset, file_list, c_i):

    print('loading data')
    g_list = []
    feat_dict = {}

    file_node = dataset[c_i]# 点的特征

    file_edge = dataset[-1]# 拓扑结构

    with open(file_list, 'r') as f:
        num_list = f.readline().strip().split()

    for i in range(len(num_list)):
        name_file_node = (file_node + '%d.txt' % int(num_list[i]))
        name_file_edge = (file_edge + '%d.txt' % int(num_list[i]))
        g = nx.Graph()
        node_features = []
        with open(name_file_node, 'r') as f:
            n_node = int(f.readline().strip())
            for j in range(n_node):
                g.add_node(j)
                row = f.readline().strip().split()
                # row_new = [(row_i) for row_i in row]
                attr = np.array(row, dtype=np.float32)
                node_features.append(attr)

        with open(name_file_edge, 'r') as f:
            n_edge = int(f.readline().strip())
            for j in range(n_edge):
                row = f.readline().strip().split()
                # 点的索引需要减去1
                g.add_edge(int(row[0]) - 1, int(row[1]) - 1)

        g_list.append(S2VGraph(g, np.array(node_features)))

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])


        g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1,0))


    print("# data: %d" % len(g_list))

    return g_list


# Generate adj for a batch graphs
def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    # Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    return Adj_block_idx_row, Adj_block_idx_cl

#Generate node id for a batch graphs
def get_graphpool(batch_graph):
    start_idx = [0]

    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

    return graph_pool.to(device)

#Get batch graphs
def get_batch_data(batch_graph):
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    X_concat = torch.from_numpy(X_concat).to(device)
    # graph-level sum pooling
    graph_pool = get_graphpool(batch_graph)

    Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(batch_graph)
    dict_Adj_block = {}
    for i in range(len(Adj_block_idx_row)):
        if Adj_block_idx_row[i] not in dict_Adj_block:
            dict_Adj_block[Adj_block_idx_row[i]] = []
        dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])

    temp = np.zeros((NUM * len(batch_graph), NUM * len(batch_graph)))
    for i in range(len(dict_Adj_block)):
        if bool(dict_Adj_block.get(i)) == True:
            temp[i][dict_Adj_block[i]] = 1
    temp = torch.tensor(temp + np.diag(np.ones(temp.shape[0])), dtype=torch.float32).to(device)
    return graph_pool, X_concat, temp