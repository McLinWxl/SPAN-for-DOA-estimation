import os
name = 'LISTA'  # LISTA, CPSS, AMI
name_train = 'LISTA-16'
name_test = 'AMI-LF4'
name_test_SNR = 'AMI-LF4'

num_layers = 16
num_layers_test = 4

config = {
    'device': 'cpu',
    'data_path': './Dataset/Data/TrainData.h5',
    'model_path': f'./Model/{name_train}/',
    'figure_path': f'./Figure/{name_train}/',
    'result_path': f'./Result/{name_train}/',
    'batch_size': 100,
    'learning_rate': 0.0004,
    'num_layers': num_layers,
    'LF': True,
    'epoch': 100,
}

config_test = {
    'device': 'cpu',
    'data_path': './Dataset/Data/TestSpectrum.h5',
    'model_path': f'./Model/{name_test}/',
    'figure_path': f'./Figure/{name_test}/',
    'result_path': f'./Result/{name_test}/',
    'batch_size': 5,
    'num_layers': num_layers_test,
    'epoch': 100,
}

config_test_SNR = {
    'device': 'cpu',
    'data_path': './Dataset/Data/TestData_varSNR_16.h5',
    'model_path': f'./Model/{name_test_SNR}/',
    'figure_path': f'./Figure/{name_test_SNR}/',
    'result_path': f'./Result/{name_test_SNR}/',
    'batch_size': 5,
    'num_layers': num_layers_test,
    'epoch': 100,
}


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


make_dir(config['model_path'])
make_dir(config['figure_path'])
make_dir(config['result_path'])
make_dir(config_test['model_path'])
make_dir(config_test['figure_path'])
make_dir(config_test['result_path'])
make_dir(config_test_SNR['model_path'])
make_dir(config_test_SNR['figure_path'])
make_dir(config_test_SNR['result_path'])


