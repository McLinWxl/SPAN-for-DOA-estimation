import h5py
import numpy as np
from DoaMethods.functions import denoise_covariance, ReadModel, timer, find_peak
import DoaMethods
import torch
from rich.progress import track
import matplotlib.pyplot as plt
import itertools


class TestCurve:

    def __init__(self, dir_test, resolution=1, num_sources=2):
        test_mat = h5py.File(dir_test, 'r')
        self.label = test_mat["LabelPower"][()].reshape(*test_mat["LabelPower"].shape, 1)
        self.covariance_matrix = denoise_covariance(test_mat["CovarianceMatrix"][()]) if len(
            test_mat["CovarianceMatrix"].shape) == 3 else \
            np.array([denoise_covariance(mat) for mat in test_mat["CovarianceMatrix"]])
        self.raw_data = test_mat["RawData"][()]
        self.dictionary = test_mat['Dictionary'][()]
        self.DOA_train = test_mat["DOA_train"][()]
        # self.DOA_train = test_mat["DOA_train"][()].T.reshape(1, *test_mat["DOA_train"].shape[:3])
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
        self.num_mesh = self.label.shape[2]
        self.label = self.label.reshape(self.num_lists, self.num_id, self.num_mesh, 1)
        self.num_sensors, self.num_snapshots = self.raw_data.shape[-2::]
        self.covariance_array = self.covariance_matrix.transpose(0, 1, 3, 2).reshape(self.num_lists, self.num_id,
                                                                                     self.num_sensors ** 2, 1)
        # self.DOA_train = self.DOA_train.T.reshape(1, self.num_id, 2) if len(self.covariance_matrix.shape) == 3 else self.DOA_train

        self.angles = np.linspace(-0.5 * (self.num_mesh - 1) * resolution, 0.5 * (self.num_mesh - 1) * resolution,
                                  self.num_mesh)

    @timer
    def test_model(self, name, model_dir, **kwargs):
        device = kwargs.get('device', 'cpu')
        self.num_layers = kwargs.get('num_layers', 4)
        dictionary = torch.from_numpy(self.dictionary)
        model = ReadModel(name=name, dictionary=dictionary, num_layers=self.num_layers, device=device).load_model(model_dir)
        model.eval()
        prediction = torch.zeros((self.num_lists, self.num_id, self.num_mesh, 1))
        prediction_layers = torch.zeros((self.num_lists, self.num_id, self.num_layers, self.num_mesh, 1))
        with torch.no_grad():
            for list_idx in track(range(self.num_lists), description="Ada-LISTA"):
                for idx in range(self.num_id):
                    cor_array_item = torch.unsqueeze(torch.from_numpy(self.covariance_array[list_idx, idx]), dim=0)
                    prediction[list_idx, idx], prediction_layers[list_idx, idx] = model(cor_array_item)
                    # plt.plot(prediction[list_idx, idx].detach().numpy())
                    # plt.show()
        return prediction, prediction_layers

    def find_peak(self, predict):
        num_lists, num_id, _, _ = predict.shape
        peak = np.zeros((num_lists, num_id, 2))
        for list_idx, idx in itertools.product(range(num_lists), range(num_id)):
            peak[list_idx, idx] = find_peak(predict[list_idx, idx].reshape(1, self.num_mesh, 1), num_sources=2).reshape(
                -1)
        return peak

    def calculate_error(self, peak):
        """
        :param peak: (num_lists, num_id, 2)
        :return: error, RMSE, NMSE, prob
        """
        num_list, num_id, _ = peak.shape
        RMSE = np.zeros(num_list)
        NMSE = np.zeros(num_list)
        prob = np.zeros(num_list)
        predict_st = peak
        DOA_train_st = self.DOA_train
        error_ = np.zeros((num_list, num_id, self.num_sources))
        for snr in range(num_list):
            for itt in range(num_id):
                predict_st[snr, itt] = np.sort(peak[snr, itt])
                DOA_train_st[snr, itt] = np.sort(self.DOA_train[snr, itt])
        for snr in range(num_list):
            error = np.abs(np.sort(predict_st[snr]) - DOA_train_st[snr])
            error_[snr] = (predict_st[snr]) - DOA_train_st[snr]
            for idx in range(num_id):
                for i in range(self.num_sources):
                    prob[snr] += np.sum(error[idx, i] <= 4.4)
                    if error[idx, i] > 4.4:
                        error[idx, i] = 10
            RMSE[snr] = np.sqrt(np.mean(error ** 2))
            NMSE[snr] = 10 * np.log10(np.mean(error ** 2) / np.mean(DOA_train_st[snr] ** 2))
            prob[snr] = prob[snr] / num_id

        return error_, RMSE, NMSE, prob / self.num_sources




