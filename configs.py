import os
name = 'AMI'
num_layers = 4
config = {
    'device': 'cpu',
    'data_path': './Dataset/Data/TrainData.h5',
    'model_path': f'./Model/{name}/',
    'figure_path': f'./Figure/{name}/',
    'result_path': f'./Result/{name}/',
    'batch_size': 100,
    'learning_rate': 0.0004,
    'num_layers': num_layers,
    'LF': True,
    'epoch': 300,
}

config_test = {
    'device': 'cpu',
    'data_path': './Dataset/Data/TestSpectrum.h5',
    'model_path': f'./Model/{name}/',
    'figure_path': f'./Figure/{name}/',
    'result_path': f'./Result/{name}/',
    'batch_size': 5,
    'num_layers': num_layers,
}

if not os.path.exists(config['model_path']):
    os.makedirs(config['model_path'])

if not os.path.exists(config['figure_path']):
    os.makedirs(config['figure_path'])

if not os.path.exists(config['result_path']):
    os.makedirs(config['result_path'])
