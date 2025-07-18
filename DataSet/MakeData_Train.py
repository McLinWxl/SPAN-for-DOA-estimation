import h5py
from DoaDataGenerator.data_generator import DataGenerator
import os
import numpy as np

abs_path = os.path.abspath(os.path.dirname(__file__))

configs = {
    'dataset_path': f'{abs_path}/Data/',
    'Start': -60,
    'End': 60,
    'Interval': 1,
    'num_sensor': 8,
    'num_snapshot': 256,
}

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)

# DOA11 = []
# DOA22 = []
# k1 = np.arange(1, 41, 1)
# for i in range(len(k1)):
#     DOA1 = np.arange(configs['Start'], configs['End'] - k1[i] + 1)
#     DOA2 = np.arange(configs['Start'] + k1[i], configs['End'] + 1)
#     DOA11 = np.concatenate([DOA11, DOA1])
#     DOA22 = np.concatenate([DOA22, DOA2])
# DOAs = np.vstack([DOA11, DOA22]).T

# num_DOAs = 5000
# DOAs = np.random.uniform(-60, 60, (num_DOAs, 2))
#
# DOAs_diff = np.abs(DOAs[:, 0] - DOAs[:, 1])
# DOAs_diff = np.sort(DOAs_diff)

# DOAs = np.random.randint(configs['Start'], configs['End'], (num_DOAs, 2))
# for i in range(num_DOAs):
#     if DOAs[i, 0] == DOAs[i, 1]:
#         if np.random.rand() > 0.5:
#             DOAs[i, 1] = np.random.randint(DOAs[i, 1] + 1, configs['End'])
#         else:
#             DOAs[i, 0] = np.random.randint(configs['Start'], DOAs[i, 0])


# DOAs = []
# k = np.arange(1, 20, 1)
# step = 1
# for i in range(10):
#     for j in range(len(k)):
#         DOA1 = np.array([-60 + i * step, -60 + i * step + k[j]]).reshape(1, 2)
#         DOA2 = np.array([0, 0 + i * step + k[j]]).reshape(1, 2)
#         if i == j == 0:
#             DOAs = np.vstack([DOA1, DOA2])
#         else:
#             DOAs = np.vstack([DOAs, DOA1])
#             DOAs = np.vstack([DOAs, DOA2])

num_DOAs = 6000
DOAs = np.zeros((num_DOAs, 2))
interval_list = np.arange(3, 23, 1)
for i in range(int(num_DOAs / len(interval_list))):
    for idx, interval in enumerate(interval_list):
        DOA1 = np.random.uniform(-60+1, 60 - interval-1, 1)
        DOA2 = DOA1 + interval + np.random.uniform(-0.5, 0.5, 1)
        DOAs[i*len(interval_list)+idx] = np.array([DOA1, DOA2]).reshape(1, 2)


snr_list = [-10, -7, -4, 0, 3]
for snr in snr_list:
    DG = DataGenerator(DOAs, snr_db=snr, is_train=True, repeat=1)
    RawData_sample, Label_sample = DG.get_raw_label()
    if snr == snr_list[0]:
        RawData = RawData_sample
        Label = Label_sample
    else:
        RawData = np.vstack([RawData, RawData_sample])
        Label = np.vstack([Label, Label_sample])

with h5py.File(f'{configs["dataset_path"]}TrainData_mini.h5', 'w') as f:
    f.create_dataset('RawData', data=RawData)
    f.create_dataset('LabelPower', data=Label)
