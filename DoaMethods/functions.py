import heapq
import numpy as np
import scipy.signal
import torch
import DoaMethods


class ReadModel:
    def __init__(self, name, dictionary, num_layers, device='cpu'):
        if name == 'LISTA':
            model = (DoaMethods.UnfoldingMethods.LISTA(dictionary=dictionary, num_layers=num_layers)
                     .to(device))
        elif name == 'AMI':
            model = (DoaMethods.UnfoldingMethods.AMI_LISTA(dictionary=dictionary, num_layers=num_layers)
                     .to(device))
        elif name == 'CPSS':
            model = (DoaMethods.UnfoldingMethods.CPSS_LISTA(dictionary=dictionary, num_layers=num_layers)
                     .to(device))
        else:
            raise ValueError("No such model")
        self.model = model

    def get_model(self):
        return self.model

    def load_model(self, modelPath):
        state_dict = torch.load(modelPath, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict['model'])
        return self.model


def denoise_covariance(covariance_matrix, num_sources=2):
    """
    Minus the noise variance (estimated by the smallest eigenvalue) from the covariance matrix.
    :param covariance_matrix:
    :param num_sources:
    :return: Denoised covariance vector
    """
    nums, M, M = covariance_matrix.shape
    covariance_matrix_clean = np.zeros((nums, M, M)) + 1j * np.zeros((nums, M, M))
    for i in range(nums):
        eigvalue = np.linalg.eigvals(covariance_matrix[i])
        smallest_eigvalue = heapq.nsmallest(int(M - num_sources), eigvalue)
        noise_variance = np.mean(smallest_eigvalue)
        noise_matrix = noise_variance * np.eye(M)
        covariance_matrix_clean[i] = covariance_matrix[i] - noise_matrix
    return covariance_matrix_clean


def soft_threshold(x, theta):
    return x.sgn() * torch.nn.functional.relu(x.abs() - theta)


def support_selection(x, theta, p):
    x_abs = x.abs()
    threshold = torch.quantile(x_abs, 1 - p, dim=1, keepdims=True)
    if isinstance(p, torch.Tensor) and p.numel() > 1:
        threshold = torch.stack([threshold[i, i, 0] for i in range(p.numel())]).unsqueeze(1)
    bypass = torch.logical_and(x_abs >= threshold, x_abs >= theta).detach()
    output = torch.where(bypass, x, soft_threshold(x, theta))
    return output


def min_max_norm(x):
    x += 1e-8
    if not np.max(x) - np.min(x):
        return np.zeros(x.shape)
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def find_peak(spectrum, num_sources=2, height_ignore=0, start_bias=60):
    numTest, num_mesh, _ = spectrum.shape
    angles = np.zeros((num_sources, numTest))
    for num in range(numTest):
        li = spectrum[num, :].reshape(-1)
        angle = np.zeros(num_sources) - 5
        peaks_idx = np.zeros(num_sources)
        grids_mesh = np.arange(num_mesh) - start_bias
        peaks, _ = scipy.signal.find_peaks(li, height=height_ignore)
        max_spectrum = heapq.nlargest(num_sources, li[peaks])
        for i in range(len(max_spectrum)):
            peaks_idx[i] = np.where(li == max_spectrum[i])[0][0]
            angle[i] = (
                li[int(peaks_idx[i] + 1)] / (li[int(peaks_idx[i] + 1)]
                                             + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] + 1)]
                + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] + 1)]
                                           + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
                if li[int(peaks_idx[i] - 1)] < li[int(peaks_idx[i] + 1)]
                else li[int(peaks_idx[i] - 1)] / (li[int(peaks_idx[i] - 1)]
                                                  + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] - 1)]
                     + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] - 1)]
                                                + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
            )
        angles[:, num] = angle.reshape(-1)
    return np.sort(angles, axis=0)[::-1]
