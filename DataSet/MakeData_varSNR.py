import h5py
from DoaDataGenerator.data_generator import DataGenerator
import os
import numpy as np
abs_path = os.path.abspath(os.path.dirname(__file__))

interval = '16'

configs = {
    'dataset_path': f'{abs_path}/Data/',
    'Start': -60,
    'End': 60,
    'Interval': 1,
    'num_sensor': 8,
    'num_snapshot': 256,
    'MC': 500
    }

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)

DOAs = np.array([
     [-16.5, 15.5],
     ])

snr_db = np.arange(-12, 13, 1)
RawData = np.zeros((len(snr_db), len(DOAs) * configs['MC'], configs['num_sensor'], configs['num_snapshot']), dtype=np.complex64)
Label = np.zeros((len(snr_db), len(DOAs) * configs['MC'], num_meshes, 1), dtype=np.float32)
for i in range(len(snr_db)):
    DG = DataGenerator(DOAs, is_train=False, snr_db=snr_db[i], repeat=configs['MC'])
    RawData[i], Label[i] = DG.get_raw_label()

with h5py.File(f'{configs["dataset_path"]}TestData_varSNR_{interval}.h5', 'w') as f:
    f.create_dataset('RawData', data=RawData)
    f.create_dataset('LabelPower', data=Label)

