import os.path
import sys
import pathlib
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime
import csv

import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv

sys.path.append('/panfs/jay/groups/32/kumarv/sharm636/Hydrophilicity/MLcode/HydroPred/')
from hydropred.models.GraphModels import *
from hydropred.utils.GraphLib import *
from hydropred.utils.config import * #surfaces_path, methane_path, phenol_path, benzene_path, ammonia_path, lattice_spacing, num_bins
from hydropred.utils.distance_lib import d0_features 

file_paths = {
    'ammonia' : ammonia_path,
    'benzene' : benzene_path,
    'phenol' : phenol_path,
    'methane' : methane_path
}

parser = argparse.ArgumentParser(description='details',
        usage='use "%(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-model', nargs="+", type=str, required=True, help='Model type (GCN,GNN,SGNN), GNN=GraphSage, SGNN=Stochastic Graph Sage')
parser.add_argument('-train_data', nargs="+", type=str, required=True, help='Training Data Solute (Ammonia, Phenol, Benzene, Methane)')
parser.add_argument('-test_data', nargs="+", type=str, required=True, help='Testing Data Solute (Ammonia, Phenol, Benzene, Methane)')

model_list = ['GCN', 'GNN', 'SGNN', 'GNN3', 'ASGNN']
model_type = ''
if len(sys.argv)>1:
    args = parser.parse_args()

    # type of model to generate
    if args.model is not None:
        model_type = args.model[0]
        print()
        # validate
        if model_type not in model_list:
            print('Model type must be one of ',model_list)
            quit()
    else:
        print('Model type required - must be one of ',model_list)
        quit()

    # get input data
    if args.train_data is not None:
        train_solute = args.train_data[0]
    else:
        print('Train Solute required.')
        quit()

    if args.test_data is not None:
        test_solute = args.test_data[0]
    else:
        print('Test Solute required.')
        quit()
        
# SURFACES
surfaces = np.load(os.path.join(surfaces_path))
# one_hot = np.load('../../data/one_hot.npy')
one_hot = np.load('../../data/char.npy')
# (993, 1093, 1193, 2193)
# b, a, p, m
OHE = {'benzene':one_hot[:993],
       'ammonia':one_hot[993:1093],
       'phenol':one_hot[1093:1193],
       'methane':one_hot[1193:2193]
      }
if train_solute == test_solute:
    data_file = pd.read_csv(os.path.join(file_paths[train_solute]))
    idx = data_file.structure_index
    surfaces_subset = np.expand_dims(surfaces[idx], axis=1)
    energy = np.expand_dims(data_file.dGsolv.to_numpy(), axis=1)

    strata = np.histogram(energy, bins=num_bins)[1]
    binned = np.digitize(energy, bins=strata[1:-1])
    test_split = 1-split_ratio

    for i in range(num_bins):
        holder = np.where(binned==i)[0]
        train_p, test_p, train_e, test_e = train_test_split(surfaces_subset[holder],energy[holder],
                                                            test_size = test_split, train_size=split_ratio, shuffle=True)
        if i==0:
            train_polarity = train_p
            test_polarity = test_p
            train_energy = train_e
            test_energy = test_e
        else:
            train_polarity = np.concatenate((train_polarity, train_p))
            test_polarity = np.concatenate((test_polarity, test_p))
            train_energy = np.concatenate((train_energy, train_e))
            test_energy = np.concatenate((test_energy, test_e))
        train_p, train_e, test_p, test_e = train_polarity, train_energy, test_polarity, test_energy
        

else:
    train_data = pd.read_csv(os.path.join(file_paths[train_solute]))
    idx = train_data.structure_index
    train_p = np.expand_dims(surfaces[idx], axis=1)
    train_e = np.expand_dims(train_data.dGsolv.to_numpy(), axis=1)

    test_data = pd.read_csv(os.path.join(file_paths[test_solute]))
    idx = test_data.structure_index
    test_p = np.expand_dims(surfaces[idx], axis=1)
    test_e = np.expand_dims(test_data.dGsolv.to_numpy(), axis=1)
    
    train_o, test_o = OHE[train_solute], OHE[test_solute]
    

train_polarity_graph = np.transpose(train_p, (0,2,3,1))
train_polarity_graph = np.concatenate([train_polarity_graph[:,i] for i in range(train_polarity_graph.shape[1])], axis=1)
test_polarity_graph = np.transpose(test_p, (0,2,3,1))
test_polarity_graph = np.concatenate([test_polarity_graph[:,i] for i in range(test_polarity_graph.shape[1])], axis=1)


adjacency = create_adjacency_matrix(num_atoms_row, num_atoms_col)
edge_index = adjacency.nonzero().t().contiguous()

# CONVERT TO TORCH TENSOR

train_polarity_graph = torch.from_numpy(train_polarity_graph.astype('f'))
test_polarity_graph = torch.from_numpy(test_polarity_graph.astype('f'))
train_energy_graph = torch.from_numpy(train_e.astype('f'))
test_energy_graph = torch.from_numpy(test_e.astype('f'))
train_o, test_o = torch.from_numpy(train_o.astype('f')), torch.from_numpy(test_o.astype('f'))
# edge_index = torch.from_numpy(edge_index.astype(int))

# ADD FEW SHOTS
indices = torch.randperm(len(test_polarity_graph))[:num_shots]
test_indices = torch.tensor([i for i in range(len(test_polarity_graph)) if i not in indices])
fewshot_features = test_polarity_graph[indices]
fewshot_energy = test_energy_graph[indices]
fewshot_OHE = test_o[indices]
# train_polarity_graph = torch.cat([train_polarity_graph, fewshot_features], dim=0)
# train_energy_graph = torch.cat([train_energy_graph, fewshot_energy], dim=0)
# train_o = torch.cat([train_o, test_o[indices]], dim = 0)
test_polarity_graph = test_polarity_graph[test_indices]
test_energy_graph = test_energy_graph[test_indices]
test_o = test_o[test_indices]

## MODEL TRAINING
num_features = train_polarity_graph.shape[-1]  # Polarity feature

model = GCN_OH(num_features, hidden_channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr_choice)

fig, axes = plt.subplots(ncols = 3, nrows = 1, figsize = (16, 5))

# Training the model
checked = 0
tol = 1e-7
min_epochs = 100
start = time.time()
for epoch in range(num_epochs):
    # Forward pass
    train_outputs = model(train_polarity_graph, train_o, edge_index)
    train_loss = criterion(train_outputs, train_energy_graph)

    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # check if we should stop
    if early_stop:
        if epoch==min_epochs:
            early_stop_loss = train_loss.item()
            early_stop_model = model
            early_stop_epoch = epoch
            saved_train_outputs = train_outputs.cpu().detach().numpy()
            try:
                early_stop_r2 = sm.r2_score(train_energy_graph.cpu().detach().numpy(), saved_train_outputs)
            except:
                early_stop_r2 = np.NaN
        else:
            if epoch > min_epochs:
                if (train_loss.item() >= early_stop_loss-tol):
                    # loss is increasing or staying the same
                    checked +=1
                else:
                    checked = 0
                    early_stop_loss = train_loss.item()
                    saved_train_outputs = train_outputs.cpu().detach().numpy()
                    try:
                        early_stop_r2 = sm.r2_score(train_energy_graph.cpu().detach().numpy(), train_outputs.cpu().detach().numpy())
                    except:
                        early_stop_r2 = np.NaN
                    early_stop_model = model
                    early_stop_epoch = epoch

                if checked > patience and early_stop_loss<1:
                    #print(f'Early stopping at Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.item():.4f}')
                    #print('Early stop epoch = ',early_stop_epoch, 'Early stop loss=', early_stop_loss)
                    stopped = True
                    break

    # Print progress
    #if epoch%200==0:
    #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.item():.4f}')

    #training time
    train_time = time.time()-start

print('BEFORE FSL')

print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.item():.4f}')
print(f'Train time: {train_time}')


# test the model 
with torch.no_grad():
    test_outputs = model(test_polarity_graph, test_o, edge_index)
    test_loss = criterion(test_outputs, test_energy_graph)
    try:
        saved_test_outputs = test_outputs.cpu().detach().numpy()
        test_r2 = sm.r2_score(test_energy_graph.cpu().detach().numpy(), saved_test_outputs)
    except:
        test_r2 = np.NaN
print(f'Test  Loss: {test_loss.item():.4f}')
print(f'Test  R2: {test_r2:.4f}')

axes[0].scatter(train_outputs.detach().numpy(), train_energy_graph.detach().numpy(), label = 'Train')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Observed')
axes[0].set_title('Train: {}'.format(train_solute))
### FEW SHOT LEARNING

optimizer = optim.Adam(model.parameters(), lr=0.1)


start = time.time()
for epoch in range(num_fewshot_epochs):
    # Forward pass
    train_outputs = model(fewshot_features, fewshot_OHE, edge_index)
    train_loss = criterion(train_outputs, fewshot_energy)

    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


    train_time = time.time()-start

print('AFTER FSL')
print(f'Epoch [{epoch+1}/{num_fewshot_epochs}], Loss: {train_loss.item():.4f}')
print(f'Train time: {train_time}')

axes[1].scatter(train_outputs.detach().numpy(), fewshot_energy.detach().numpy(), label = 'k Shots')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Observed')
axes[1].set_title('Few Shot: {}'.format(test_solute))
axes[1].legend(loc='upper left')


# compute train r2
if not early_stop:
    saved_train_outputs = train_outputs.cpu().detach().numpy()
    try:
        train_r2 = sm.r2_score(train_energy_graph.cpu().detach().numpy(), saved_train_outputs)
    except:
        train_r2 = np.NaN

# test the model 
with torch.no_grad():
    test_outputs = model(test_polarity_graph, test_o, edge_index)
    test_loss = criterion(test_outputs, test_energy_graph)
    try:
        saved_test_outputs = test_outputs.cpu().detach().numpy()
        test_r2 = sm.r2_score(test_energy_graph.cpu().detach().numpy(), saved_test_outputs)
    except:
        test_r2 = np.NaN

print(f'Test  Loss: {test_loss.item():.4f}')
print(f'Test  R2: {test_r2:.4f}')

axes[2].scatter(test_outputs.detach().numpy(), test_energy_graph.detach().numpy(), label = 'Test')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Observed')
axes[2].set_title('Test: {}'.format(test_solute))
axes[2].legend(loc='upper left')

fig.savefig('../../img/FS_train_{}_test_{}'.format(train_solute, test_solute))