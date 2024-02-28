surfaces_path = '/scratch.global/sbinkash/generative_polymer_interfaces/generative_polymer_interfaces/md_emulator/data/surfaces_for_training.npy'
methane_path = '/scratch.global/sbinkash/generative_polymer_interfaces/generative_polymer_interfaces/md_simulations/data/hexagonal_surfaces_methane/MD_simulations/dGSolvData.csv'
phenol_path = '/scratch.global/sbinkash/generative_polymer_interfaces/generative_polymer_interfaces/md_simulations/data/hexagonal_surfaces_phenol/MD_simulations/dGSolvData.csv'
benzene_path = '/scratch.global/sbinkash/generative_polymer_interfaces/generative_polymer_interfaces/md_simulations/data/hexagonal_surfaces_benzene/MD_simulations/dGSolvData.csv'
ammonia_path = '/scratch.global/sbinkash/generative_polymer_interfaces/generative_polymer_interfaces/md_simulations/data/hexagonal_surfaces_ammonia/MD_simulations/dGSolvData.csv'

num_atoms_row, num_atoms_col = 8, 6
lattice_spacing = 0.497
num_bins=4
split_ratio = 0.5

num_shots = 5

hidden_channels = 64
lr_choice = 0.1
early_stop = False
num_epochs = 1000
num_fewshot_epochs = 500