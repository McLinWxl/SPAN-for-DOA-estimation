import h5py
from DoaDataGenerator.data_generator import DataGenerator
import os
import numpy as np
abs_path = os.path.abspath(os.path.dirname(__file__))

interval = '26'

configs = {
    'dataset_path': f'{abs_path}/Data/',
    'Start': -60,
    'End': 60,
    'Interval': 1,
    'num_sensor': 8,
    'SNR': 0,
    'MC': 1000
    }

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)

if interval == '26':
    DOAs = np.array([
         [-13.5, 12.5],
         ])
elif interval == '6':
    DOAs = np.array([
         [-3.5, 2.5],
         ])
elif interval == '16':
    DOAs = np.array([
         [-8.5, 7.5],
         ])
num_snapshots = np.arange(10, 410, 10)
max_snapshot = np.max(num_snapshots)
RawData = np.zeros((len(num_snapshots), len(DOAs) * configs['MC'], configs['num_sensor'], max_snapshot), dtype=np.complex64)
Label = np.zeros((len(num_snapshots), len(DOAs) * configs['MC'], num_meshes, 1), dtype=np.float32)
for i in range(len(num_snapshots)):
    DG = DataGenerator(DOAs, is_train=False, snr_db=configs['SNR'], repeat=configs['MC'], num_snapshot=num_snapshots[i])
    raw_data, Label[i] = DG.get_raw_label()
    RawData[i, :, :, :num_snapshots[i]] = raw_data

with h5py.File(f'{configs["dataset_path"]}TestData_varSnapshots_{interval}.h5', 'w') as f:
    f.create_dataset('RawData', data=RawData)
    f.create_dataset('LabelPower', data=Label)

