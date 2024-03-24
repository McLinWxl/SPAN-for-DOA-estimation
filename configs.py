import os
abs_path = os.path.abspath(os.path.dirname(__file__))
# abs_path = os.path.dirname(abs_path)
name = 'ALISTA-SS'
is_insert_superresolution = True
# is_insert_superresolution = True if name == 'ALISTA-SS' else False
# LISTA, CPSS, AMI, ALISTA
# DCNN
# MUSIC, MVDR, SBL, ISTA
###################
num_sensors = 8
# For LISTA, CPSS, AMI
is_LF = True

num_layers = 60
num_layers_saved = 60
num_layers_test = 60

is_checkpoint = False
####################
mode = 'SNR'  # SNR, Snapshots, Separation, Sensors
testSNR_interval = 35
####################
batch_size = 128
lr = 0.001
epoch = 600
is_scheduler = True
warmup_epoch = 400
####################
UnfoldingMethods= ['LISTA', 'CPSS', 'AMI', 'ALISTA', 'ALISTA-SS']
DataMethods = ['DCNN']
ModelMethods = ['MUSIC', 'MVDR', 'SBL', 'ISTA']
####################
if name in DataMethods or name in ModelMethods:
    name_train = f'{name}'
    name_test = f'{name}'
    name_test_SNR = f'{name}'
elif name in UnfoldingMethods:
    if is_LF:
        name_train = f'{name}-LF{num_layers}'
        name_test = f'{name}-LF{num_layers_saved}'
        name_test_SNR = f'{name}-LF{num_layers_saved}'
    else:
        name_train = f'{name}-{num_layers}'
        name_test = f'{name}-{num_layers_saved}'
        name_test_SNR = f'{name}-{num_layers_saved}'
else:
    raise ValueError("Wrong name!")

config = {
    'device': 'cpu',
    'data_path': f'{abs_path}/DataSet/Data/TrainData_mini.h5',
    'model_path': f'{abs_path}/Model_mini/{name_train}/',
    'figure_path': f'{abs_path}/Figure_mini/{name_train}/',
    'result_path': f'{abs_path}/Result_mini/{name_train}/',
    'batch_size': batch_size,
    'learning_rate': lr,
    'num_layers': num_layers,
    'LF': is_LF,
    'epoch': epoch,
    'scheduler': is_scheduler,
    'warmup_epoch': warmup_epoch,
}

config_test = {
    'device': 'cpu',
    'data_path': f'{abs_path}/DataSet/Data/TestSpectrum_{num_sensors}.h5',
    'model_path': f'{abs_path}/checkpoint/{name_test}.pth' if is_checkpoint else f'{abs_path}/Model_mini/{name_test}/',
    'figure_path': f'{abs_path}/Figure_mini/{name_test}/',
    'result_path': f'{abs_path}/Result_mini/{name_test}/',
    'batch_size': 1,
    'num_layers': num_layers_test,
}

if mode in ['SNR', 'Snapshots']:
    data_path_static = f'{abs_path}/DataSet/Data/TestData_var{mode}_{testSNR_interval}_{num_sensors}.h5'
elif mode == 'Sensors':
    data_path_static = f'{abs_path}/DataSet/Data/TestData_varSensors_{testSNR_interval}.h5'
elif mode == 'Separation':
    data_path_static = f'{abs_path}/DataSet/Data/TestData_varSeparation.h5'
else:
    raise ValueError("Wrong mode!")


config_test_static = {
    'mode': mode, # SNR, Snapshots, Separation
    'testSNR_interval': testSNR_interval,
    'device': 'cpu',
    'data_path': data_path_static,
    'model_path': f'{abs_path}/checkpoint/{name_test_SNR}.pth' if is_checkpoint else f'{abs_path}/Model_mini/{name_test_SNR}/',
    'figure_path': f'{abs_path}/Figure_mini/{name_test_SNR}/',
    'result_path': f'{abs_path}/Result_mini/{name_test_SNR}/',
    'batch_size': 1,
    'num_layers': num_layers_test,
}


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


make_dir(config['model_path'])
make_dir(config['figure_path'])
make_dir(config['result_path'])
# make_dir(config_test['model_path'])
make_dir(config_test['figure_path'])
make_dir(config_test['result_path'])
# make_dir(config_test_static['model_path'])
make_dir(config_test_static['figure_path'])
make_dir(config_test_static['result_path'])