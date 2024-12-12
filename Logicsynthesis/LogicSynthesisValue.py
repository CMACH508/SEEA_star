import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv,GINConv,GATConv,TAGConv
from torch_geometric.nn import global_mean_pool, global_max_pool, SAGPooling, TopKPooling, ASAPooling, global_add_pool
from torch_geometric.nn.norm import BatchNorm, GraphNorm, LayerNorm, InstanceNorm
import torch.nn.functional as F
import torch_geometric
from torch_sparse import SparseTensor
import abc_py as abcPy
from transformers import BertTokenizer, BertModel


allowable_features = {
    'node_type': [0, 1, 2],
    'num_inverted_predecessors': [0, 1, 2]
}


NUM_LENGTH_EPISODES = 10
RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
BERT_MODEL_NAME = './bert_base_cased'
tz = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))


full_node_feature_dims = get_node_feature_dims()


class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        x_embedding = self.node_type_embedding(x[:, 0])
        x_embedding = torch.cat((x_embedding, x[:, 1].reshape(-1, 1)), dim=1)
        return x_embedding


class AIGEncoder(torch.nn.Module):
    def __init__(self, node_encoder, input_dim, num_layer=2, emb_dim=128, gnn_type='gcn', norm_type='batch',
                 final_layer_readout=True,
                 pooling_type=None, pooling_ratio=0.8, readout_type=['max', 'sum']):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(AIGEncoder, self).__init__()
        self.num_layer = num_layer
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder
        self.gnn_conv = GCNConv
        self.norm_type = BatchNorm
        self.isPooling = False if pooling_type == None else True
        self.pooling_ratio = pooling_ratio
        self.final_layer_readout = final_layer_readout

        ### Select the type of Graph Conv Networks
        if gnn_type == 'gin':
            self.gnn_conv = GINConv
        elif gnn_type == 'gat':
            self.gnn_conv = GATConv
        elif gnn_type == 'tag':
            self.gnn_conv = TAGConv

        ### Select the type of Normalization
        if norm_type == 'graph':
            self.norm_type = GraphNorm
        elif norm_type == 'layer':
            self.norm_type = LayerNorm
        elif norm_type == 'instance':
            self.norm_type = InstanceNorm

        ## Pooling Layers
        if pooling_type == 'topk':
            self.pool_type = TopKPooling
        elif pooling_type == 'sag':
            self.pool_type = SAGPooling
        elif pooling_type == 'asap':
            self.pool_type = ASAPooling

        ###List of GNNs and layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        if self.isPooling:
            self.pools = torch.nn.ModuleList()

        ## First layer
        self.convs.append(self.gnn_conv(input_dim, emb_dim))
        self.norms.append(self.norm_type(emb_dim))
        if self.isPooling:
            self.pools.append(self.pool_type(emb_dim))

        ## Intermediate Layers
        for _ in range(1, num_layer - 1):
            self.convs.append(self.gnn_conv(emb_dim, emb_dim))
            self.norms.append(self.norm_type(emb_dim))
            if self.isPooling:
                self.pools.append(self.pool_type(in_channels=emb_dim, ratio=self.pooling_ratio))

        ## Last Layer
        self.convs.append(self.gnn_conv(emb_dim, emb_dim))
        self.norms.append(self.norm_type(emb_dim))

        ## Global Readout Layers
        self.readout = []
        for readoutConfig in readout_type:
            if readoutConfig == 'max':
                self.readout.append(global_max_pool)
            elif readoutConfig == 'mean':
                self.readout.append(global_mean_pool)
            elif readoutConfig == 'sum':
                self.readout.append(global_add_pool)

    def forward(self, batched_data):
        edge_index = batched_data.edge_index
        batch = batched_data.batch
        #size = torch.Size([torch.max(edge_index) + 1, torch.max(edge_index) + 1])
        #adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=size)
        adj = SparseTensor(row=edge_index[0], col=edge_index[1])
        x = torch.cat([batched_data.node_type.reshape(-1, 1), batched_data.num_inverted_predecessors.reshape(-1, 1)], dim=1)
        h = self.node_encoder(x)
        finalReadouts = []
        for layer in range(self.num_layer):
            h = self.convs[layer](h, adj.t())
            if layer != self.num_layer - 1:
                h = F.relu(h)
                if self.isPooling:  # Not pooling in the last layer
                    poolOutput = self.pools[layer](h, edge_index=edge_index, batch=batch)
                    h, edge_index, batch = poolOutput[0], poolOutput[1], poolOutput[3]
                if self.final_layer_readout:
                    continue
            finalReadouts.append(self.readout[0](h, batch))
            finalReadouts.append(self.readout[1](h, batch))
        aigEmbedding = torch.cat(finalReadouts, dim=1)
        aigEmbedding = torch.round(aigEmbedding, decimals=3)
        return aigEmbedding


class LogicSynthesisValue(nn.Module):
    def __init__(self, init_graph_data=True, node_enc_outdim=3, gnn_hidden_dim=32, num_gcn_layer=2,
                 gnn_type='gcn', norm_type='batch', final_layer_readout=True,
                 pooling_type=None, pooling_ratio=0.8, readout_type=['mean', 'max'], n_hidden=256, n_actions=7):
        super(LogicSynthesisValue, self).__init__()
        self.init_graph_data = init_graph_data
        if self.init_graph_data:
            self.node_encoder = NodeEncoder(emb_dim=node_enc_outdim)
            self.aig = AIGEncoder(self.node_encoder, input_dim=node_enc_outdim + 1, num_layer=num_gcn_layer,
                                  emb_dim=gnn_hidden_dim, gnn_type=gnn_type,
                                  norm_type=norm_type, final_layer_readout=final_layer_readout,
                                  pooling_type=pooling_type, pooling_ratio=pooling_ratio, readout_type=readout_type)
            self.aig_emb_dim = num_gcn_layer * gnn_hidden_dim * len(readout_type)
            if final_layer_readout == True:
                self.aig_emb_dim = gnn_hidden_dim * len(readout_type)
            self.aig_emb_dim += 768
        else:
            self.aig_emb_dim = 768

        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.denseLayer = nn.Linear(self.aig_emb_dim, n_hidden)
        self.dense_v1 = nn.Linear(n_hidden, n_hidden)
        self.dense_v2 = nn.Linear(n_hidden, 1)
        torch.nn.init.kaiming_uniform_(self.dense_v2.weight.data)
        torch.nn.init.kaiming_uniform_(self.dense_v1.weight.data)
        torch.nn.init.kaiming_uniform_(self.denseLayer.weight.data)

    def forward(self, batchData, bert_model, feature=False):
        seqEmbedding = bert_model(batchData.data_input, batchData.data_attention)
        seqEmbedding = seqEmbedding.pooler_output
        init_aig_embedding = self.aig(batchData)
        finalEmbedding = torch.cat([init_aig_embedding, seqEmbedding], dim=1)
        aigFCOutput = F.leaky_relu(self.denseLayer(finalEmbedding))
        v1Out = F.leaky_relu(self.dense_v1(aigFCOutput))
        value = torch.tanh(self.dense_v2(v1Out)).view(-1)
        if feature:
            return value, torch.sigmoid(v1Out)
        else:
            return value


class LogicSynthesisDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        input = self.data['input'][item]
        target = self.data['target'][item]
        return {
            'input': input,
            'target': target
        }

    def __len__(self):
        return len(self.data['target'])


def extract_init_graph(state):
    _abc = abcPy.AbcInterface()
    _abc.start()
    _abc.read(state)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes
    return data


def extract_pytorch_graph(depth, actionSequence, graphData=None):
    seqSentence = str(depth) + " " + actionSequence
    data = createDataFromGraphAndSequence(seqSentence, graphData)
    return data


def createDataFromGraphAndSequence(seqSentence, graphData=None):
    encoded = tz.encode_plus(
        text=seqSentence,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=32,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
    )
    data = {}
    data['data_input'] = encoded['input_ids']
    data['data_attention'] = encoded['attention_mask']
    if not graphData == None:
        for k, v in graphData.items():
            data[k] = v
        numNodes = data['nodes']
        data = torch_geometric.data.Data.from_dict(data)
        data.num_nodes = numNodes
    else:
        data = torch_geometric.data.Data.from_dict(data)
    return data






