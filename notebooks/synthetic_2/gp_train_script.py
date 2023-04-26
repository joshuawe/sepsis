from afa.configurations.utils_ts import specify_default_paths_ts
from afa.data_modelling.datasets.data_loader.data_loader_ts import DataLoader_ts
from afa.data_modelling.missingness.multiple_imputation.multiple_imputation_model_ts import MultipleImputationModel_ts



# which dataset to work on 
dataset_name   = "synthetic_2"

# name for of missingness scenario 
miss_scenario  = 'MCAR_1'

# automatically specify some path locations (change paths manually if needed) 
paths = specify_default_paths_ts(dataset_name = dataset_name , miss_scenario = miss_scenario) 

# name for ps_model 
mi_model_name  = 'mi_simple'
mi_model_name  = 'mi_simple_gp'

# new (where to save the model) 
mi_model_dir = paths['data_dir']  + 'mi_models' + '/' + mi_model_name + '/'

device = 'cuda'  # 'cuda' or 'cpu'

# Config for dataset preparation (torch.DataSet class)
dataset_params = {
    'missingness_value': 'nan',   # int, float or 'nan'
    'missingness_rate': (0, 0),
    'device': device  # 'cuda' or 'cpu'
}

# Config for dataloader (torch.DataLoader class)
dataloader_params = {
    'batch_size': 400, 
    'shuffle': False, 
    # 'prefetch_factor': 1, # increase for speed up
    # 'num_workers': 0,     # set higher for faster throughput
    'drop_last': True
}

# Config for trainer (pytorch_lightning.Trainer class)
trainer_params = {
    'max_epochs': 250,    # number of epochs to train
    'auto_lr_find': False,
    'fast_dev_run': False,
    'accelerator': device,
    'devices': 1,
    'profiler': None,  # 'simple', 'advanced', None
    'num_sanity_val_steps': 0
}

# Config for gp_model (GPImputer class)
gp_params = {
    'model_type': 'gaussian_process',
    'dataset_params' : dataset_params,
    'dataloader_params' : dataloader_params,
    'trainer_params' : trainer_params,
    'num_tasks': 5,  # number of tasks == number of features
    'num_kernels': 10,
    'data_mode': 'no_simulation',   # 'no_simulation' or 'simulation', with simulation a ground truth is expected to passed as well, ground truth = values for data that is missing in train dataloader
    # 'ckpt_path': 'best_model-v_recon_loss_target=1.10-epoch=142.ckpt',  # path to checkpoint of trained model, full path or relative to model directory    
}
    
# Config for mi_model from AFA module (MultipleImputationModel_ts class)
mi_model_params = {
    'name' : mi_model_name, 
    'directory' : mi_model_dir,
    'base_model_params' : gp_params
}


data_loader = DataLoader_ts(data_file                  = paths['data_file'],
                            temporal_data_file         = paths['temporal_data_file'],
                            superfeature_mapping_file  = paths['superfeature_mapping_file'],
                            problem_file               = paths['problem_file'],
                            afa_problem_files          = paths['afa_problem_files'], 
                            miss_model_files           = paths['miss_model_files'], 
                            folds_file                 = paths['folds_file'] )
dataset = data_loader.load()



mi_model = MultipleImputationModel_ts(  
                name                         = mi_model_params['name'], 
                m_graph                      = dataset.miss_model.m_graph, 
                superfeature_mapping         = dataset.superfeature_mapping,
                target_superfeature_names    = dataset.afa_problem.target_superfeature_names,
                model_params                 = mi_model_params,
                directory                    = mi_model_params['directory']) 


mi_model.fit(dataset, fold=None, train_split='train', valid_split='val', fit_again=False)
