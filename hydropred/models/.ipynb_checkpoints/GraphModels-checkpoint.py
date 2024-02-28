import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv,SAGEConv, GraphSAGE, GraphConv
import torch_geometric.nn as geom_nn

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        x = x.mean(dim=1)
        #x = x.sum(dim=1)
        return x

    
class GCN_OH(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN_OH, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.linear = nn.Linear(4, 1)

    def forward(self, x, o, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        x = x.mean(dim=1)
        ohe = self.linear(o)
        x = (x + ohe) #* 0.5
        #x = x.sum(dim=1)
        return x


# GraphSAGEConv
class GNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        x = x.mean(dim=1)
        #x = x.sum(dim=1)
        return x
    
class GNN3(nn.Module):
    def __init__(self, num_features, hidden_channels, version=1):
        super(GNN3, self).__init__()
        if version == 1:
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels,1)

        elif version == 2:
            self.conv1 = SAGEConv(num_features, hidden_channels//2)
            self.conv2 = SAGEConv(hidden_channels//2, hidden_channels//4)
            self.conv3 = SAGEConv(hidden_channels//4,1)
        
        elif version == 3:
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels//2)
            self.conv3 = SAGEConv(hidden_channels//2,1)
        
        elif version == 4:
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels//4)
            self.conv3 = SAGEConv(hidden_channels//4,1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        
        x = x.mean(dim=1)
        #x = x.sum(dim=1)
        return x
    

# Stochastic GraphSAGE
class SGNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(SGNN, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        # Mean & Log variance
        self.mu_transform = nn.Linear(1, 1)
        self.logvar_transform = nn.Linear(1, 1)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        x = x.mean(dim=1)
        #x = x.sum(dim=1)
        mu, logvar = self.mu_transform(x), self.logvar_transform(x)
        std = torch.exp(logvar)
        z = mu + torch.randn_like(std)*std
        return z

# stochastic graph sage with attention
class ASGNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(ASGNN, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        # Mean & Log variance
        self.mu_transform = nn.Linear(1, 1)
        self.logvar_transform = nn.Linear(1, 1)
        self.attention = SAGEConv(1, 1)#nn.Linear(1, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        att = self.attention(x, edge_index)
        att = att / x.shape[1]
        att = nn.functional.softmax(att)
        self.att = att
        x = x*att #+x
        x = x.mean(dim=1)
        # use sum here for consistency with others
        #x = x.sum(dim=1)
        mu, logvar = self.mu_transform(x), self.logvar_transform(x)
        std = torch.exp(logvar)
        x = mu + torch.randn_like(std)*std
        return x
    

    class GraphConvModel(nn.Module):
        def __init__(self,in_channels, out_channels, aggr, bias=True):
            super(GraphConvModel, self).__init__()
            #self.conv1 = GraphConv(in_channels, out_channels, aggr, bias)
            num_layers = 2

            layers = []
        
            for l_idx in range(num_layers - 1):
                layers += [
                    GraphConv(in_channels=in_channels, out_channels=out_channels, aggr=aggr, bias=bias),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                ]
                in_channels = c_hidden
            layers += [GraphConv(in_channels=in_channels, out_channels=out_channels, aggr=aggr, bias=bias)]
            self.layers = nn.ModuleList(layers)

        def forward(self, x, edge_index):
            for layer in self.layers:
                if isinstance(layer, geom_nn.MessagePassing):
                    x = layer(x, edge_index)
                else:
                    x = layer(x)
            return x
        
