import h5py
import numpy as np
from functions import denoise_covariance, ReadModel
import DoaMethods


class TestCurve:

    def __init__(self, dir_test, resolution=1, num_sources=2):
        test_mat = h5py.File(dir_test, 'r')
        self.label = test_mat["LabelPower"][()].reshape(*test_mat["LabelPower"].shape, 1)
        self.covariance_matrix = denoise_covariance(test_mat["CovarianceMatrix"][()]) if len(test_mat["CovarianceMatrix"].shape) == 3 else \
                                 np.array([denoise_covariance(mat) for mat in test_mat["CovarianceMatrix"]])
        self.raw_data = test_mat["RawData"][()]
        self.dictionary = test_mat['Dictionary'][()]
        self.DOA_train = test_mat["DOA_train"][()].T.reshape(1, *test_mat["DOA_train"].shape[:2])
        self.num_sources = num_sources

        if len(test_mat["PseudoSpectrum"].shape) == 3:
            self.PseudoSpectrum = (test_mat["PseudoSpectrum"] - np.min(test_mat["PseudoSpectrum"])) / \
                                  (np.max(test_mat["PseudoSpectrum"]) - np.min(test_mat["PseudoSpectrum"]))
            self.covariance_matrix = self.covariance_matrix.reshape(1, *self.covariance_matrix.shape)
        elif len(test_mat["PseudoSpectrum"].shape) == 4:
            self.PseudoSpectrum = (test_mat["PseudoSpectrum"] - np.min(test_mat["PseudoSpectrum"])) / \
                                  (np.max(test_mat["PseudoSpectrum"]) - np.min(test_mat["PseudoSpectrum"]))
        else:
            raise ValueError("Wrong dimension of PseudoSpectrum")

        self.num_lists, self.num_id, self.num_sensors, _ = self.covariance_matrix.shape
        _, _, self.num_mesh = self.label.shape
        self.label = self.label.reshape(self.num_lists, self.num_id, self.num_mesh, 1)
        self.num_sensors, self.num_snapshots = self.raw_data.shape[-2::]
        self.covariance_array = self.covariance_matrix.transpose(0, 2, 1).reshape(self.num_lists, self.num_id, self.num_sensors ** 2, 1)
        self.num_id = self.num_id if len(self.covariance_matrix.shape) == 3 else self.num_lists
        self.DOA_train = self.DOA_train if len(self.covariance_matrix.shape) == 3 else self.DOA_train.T.reshape(1, self.num_id, 2)

        self.angles = np.linspace(-0.5 * (self.num_mesh - 1) * resolution, 0.5 * (self.num_mesh - 1) * resolution, self.num_mesh)

    def
