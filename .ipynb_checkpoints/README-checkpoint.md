
HydroPred, an efficient PyTorch framework for Hydrophilicty Molecular Property Prediction
=========================================================================================

Four solutes:

- Methane
- Benzene
- Ammonia
- Phenol

Installation
------------

Install package requirements from requirements.txt file. The framework relies on the torch and torch geometric modules.

- torch==1.12.0
- torch_geometric @ git+https://github.com/pyg-team/pytorch_geometric.git@2febd3820ec63bd12834237b3be3453ee5b08c2e





Example
-------

In train sub-directory, you can find scripts for training and evaluating the model.

- In distribution (stratified sampling)
python3 train.py -model 'GCN' -train_data 'methane' -test_data 'methane'

- OOD
python3 train.py -model 'GCN' -train_data 'methane' -test_data 'ammonia'

- With one-hot-encoded features representing solute types
python3 train_OH.py -model 'GCN' -train_data 'methane' -test_data 'ammonia'

- Few shot learning ( number of shots defined in hydropred/utils/config.py file)
python3 train_OH_FS.py -model 'GCN' -train_data 'methane' -test_data 'ammonia'



Related Projects
----------------