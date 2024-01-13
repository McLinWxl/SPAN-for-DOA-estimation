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
    'num_snapshot': 256,
    'MC': 100
    }

Angles = np.arange(configs['Start'], configs['End'] + configs['Interval'], configs['Interval'])
num_meshes = len(Angles)

if interval == '26':
    DOAs = np.array([
         [-13.0, 13.0],
         [-13.1, 12.9],
         [-13.2, 12.8],
         [-13.3, 12.7],
         [-13.4, 12.6],
         [-13.5, 12.5],
         [-13.6, 12.4],
         [-13.7, 12.3],
         [-13.8, 12.2],
         [-13.9, 12.1],
         [-14.0, 12.0],
         ])
elif interval == '6':
    DOAs = np.array([
         [-3.0, 3.0],
         [-3.1, 2.9],
         [-3.2, 2.8],
         [-3.3, 2.7],
         [-3.4, 2.6],
         [-3.5, 2.5],
         [-3.6, 2.4],
         [-3.7, 2.3],
         [-3.8, 2.2],
         [-3.9, 2.1],
         [-4.0, 2.0],
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

