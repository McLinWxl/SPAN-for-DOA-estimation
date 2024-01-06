import h5py
import torch.utils.data
import matplotlib.pyplot as plt
from DoaMethods.functions import denoise_covariance, min_max_norm
import numpy


class MakeDataset(torch.utils.data.Dataset):
    """
    :param: h5 file path, with the following keys:
    CovarianceMatrix: (num_samples, num_sensors, num_sensors)
    LabelPower: (num_samples, num_mesh)
    Dictionary: (num_sensors**2, num_mesh)
    PseudoSpectrum: (num_samples, num_mesh, num_sources)
    :returns: [CovarianceVector[idx], Label[idx]]
    """
    def __init__(self, path):
        dataset = h5py.File(path)
        covariance_matrix = dataset['CovarianceMatrix'][()]
        label = dataset['LabelPower'][()]
        self.dictionary = dataset['Dictionary'][()]
        pseudo_spectrum = dataset['PseudoSpectrum'][()]
        # To virtualization in Paper
        # rawdata = dataset['RawData'][()]
        # plt.matshow(rawdata[0, :, 0:30].imag, cmap=plt.cm.Reds)
        # plt.matshow(covariance_matrix[0].real, cmap=plt.cm.Reds)
        # plt.matshow(covariance_matrix[0, 0:4, 0:4].reshape(-1, 1).real, cmap=plt.cm.Reds)
        # plt.matshow(np.matmul(self.dictionary.transpose(1, 0).conj(), covariance_matrix[12000].reshape(-1, 1)).real, cmap=plt.cm.Reds)
        # plt.show()
        len_dataset, num_mesh, num_sources = pseudo_spectrum.shape
        num_sensors = covariance_matrix.shape[1]

        self.label = label.reshape(len_dataset, num_mesh, 1)
        self.label /= numpy.linalg.norm(self.label, axis=1, keepdims=True)
        # self.label /= numpy.sqrt(2)
        self.covariance_matrix = denoise_covariance(covariance_matrix)
        self.covariance_vector = self.covariance_matrix.transpose(0, 2, 1).reshape(len_dataset, num_sensors ** 2, 1)

    def __getitem__(self, index):
        return self.covariance_vector[index], self.label[index]

    def get_dictionary(self):
        return self.dictionary

    def __len__(self):
        return len(self.label)

