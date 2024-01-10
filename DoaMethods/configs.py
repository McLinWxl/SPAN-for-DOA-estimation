import os
abs_path = os.path.abspath(os.path.dirname(__file__))
abs_path = os.path.dirname(abs_path)


name = 'AMI'
# LISTA, CPSS, AMI,
# DCNN
# MUSIC, MVDR, SBL, ISTA

# For LISTA, CPSS, AMI
is_LF = False
num_layers = 10
num_layers_test = 10
is_checkpoint = True
####################
mode = 'Snapshots'  # SNR, Snapshots, Separation
testSNR_interval = 6

UnfoldingMethods = ['LISTA', 'CPSS', 'AMI']
DataMethods = ['DCNN']
ModelMethods = ['MUSIC', 'MVDR', 'SBL', 'ISTA']

if name in DataMethods or name in ModelMethods:
    name_train = f'{name}'
    name_test = f'{name}'
    name_test_SNR = f'{name}'
elif name in UnfoldingMethods:
    if is_LF:
        name_train = f'{name}-LF{num_layers}'
        name_test = f'{name}-LF{num_layers}'
        name_test_SNR = f'{name}-LF{num_layers}'
    else:
        name_train = f'{name}-{num_layers}'
        name_test = f'{name}-{num_layers}'
        name_test_SNR = f'{name}-{num_layers}'
else:
    raise ValueError("Wrong name!")



config = {
    'device': 'cpu',
    'data_path': f'{abs_path}/Dataset/Data/TrainData.h5',
    'model_path': f'{abs_path}/Model/{name_train}/',
    'figure_path': f'{abs_path}/Figure/{name_train}/',
    'result_path': f'{abs_path}/Result/{name_train}/',
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_layers': num_layers,
    'LF': is_LF,
    'epoch': 800,
    'scheduler': True,
    'warmup_epoch': 450,
}

config_test = {
    'device': 'cpu',
    'data_path': f'{abs_path}/Dataset/Data/TestSpectrum.h5',
    'model_path': f'{abs_path}/checkpoint/{name_test}.pth' if is_checkpoint else f'{abs_path}/Model/{name_test}/',
    'figure_path': f'{abs_path}/Figure/{name_test}/',
    'result_path': f'{abs_path}/Result/{name_test}/',
    'batch_size': 5,
    'num_layers': num_layers_test,
    'epoch': 0,
}

config_test_static = {
    'mode': mode, # SNR, Snapshots, Separation
    'testSNR_interval': testSNR_interval,
    'device': 'cpu',
    'data_path': f'{abs_path}/Dataset/Data/TestData_var{mode}_{testSNR_interval}.h5' if mode == 'SNR' or mode == 'Snapshots' else f'{abs_path}/Dataset/Data/TestData_Separation_Ori.h5',
    'model_path': f'{abs_path}/checkpoint/{name_test_SNR}.pth' if is_checkpoint else f'{abs_path}/Model/{name_test_SNR}/',
    'figure_path': f'{abs_path}/Figure/{name_test_SNR}/',
    'result_path': f'{abs_path}/Result/{name_test_SNR}/',
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
make_dir(config_test_static['model_path'])
make_dir(config_test_static['figure_path'])
make_dir(config_test_static['result_path'])


