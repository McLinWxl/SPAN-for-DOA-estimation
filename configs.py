import os
name = 'AMI'  # LISTA, CPSS, AMI
is_LF = True
num_layers = 20
num_layers_test = 20
if is_LF:
    name_train = f'{name}-LF{num_layers}'
    name_test = f'{name}-LF{num_layers}'
    name_test_SNR = f'{name}-LF{num_layers}'
else:
    name_train = f'{name}-{num_layers}'
    name_test = f'{name}-{num_layers}'
    name_test_SNR = f'{name}-{num_layers}'

testSNR_interval = 6

config = {
    'device': 'cpu',
    'data_path': './Dataset/Data/TrainData.h5',
    'model_path': f'./Model/{name_train}/',
    'figure_path': f'./Figure/{name_train}/',
    'result_path': f'./Result/{name_train}/',
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_layers': num_layers,
    'LF': is_LF,
    'epoch': 801,
    'scheduler': True,
    'warmup_epoch': 601,
}

config_test = {
    'device': 'cpu',
    'data_path': './Dataset/Data/TestSpectrum.h5',
    'model_path': f'./Model/{name_test}/',
    'figure_path': f'./Figure/{name_test}/',
    'result_path': f'./Result/{name_test}/',
    'batch_size': 5,
    'num_layers': num_layers_test,
    'epoch': 0,
}

config_test_SNR = {
    'testSNR_interval': testSNR_interval,
    'device': 'cpu',
    'data_path': f'./Dataset/Data/TestData_varSNR_{testSNR_interval}.h5',
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


