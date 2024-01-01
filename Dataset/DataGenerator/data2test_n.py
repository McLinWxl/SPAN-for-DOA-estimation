import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../DataGeneration/')
from DataGenerator import signal_generate, feature_extract_R, add_awgn  # type: ignore


M = 8
snapshot = 256
f0 = 1e6
fc = 1e6
lam = fc / f0
fs = 4 * f0
C = M * (M - 1)
# SNR = 0

D_start = -60
D_stop = 60

DOA11 = []
DOA22 = []
times = 3000
DOA_train = np.zeros((2, times))
for idx in range(times):
    DOA_train[0, idx] = np.random.uniform(-55, 15)
    DOA_train[1, idx] = DOA_train[0, idx] + np.random.uniform(1, 40)

theta = np.arange(D_start, D_stop + 1).reshape(-1)
L = len(theta)

A = np.exp(1j * np.pi * np.arange(M)[:, np.newaxis] * np.sin(np.deg2rad(theta)) / lam)

H = np.zeros((M ** 2, L), dtype=np.complex128)

for i in range(M):
    s = np.exp(-1j * np.pi * (i) * np.sin(np.deg2rad(theta)))
    B = np.diag(s)
    fhi = np.matmul(A, B)
    H[i * M:(i + 1) * M, :] = fhi

S_label = np.zeros((DOA_train.shape[1], L))
Label_SSH = np.zeros((DOA_train.shape[1], L))
R_est = np.zeros((DOA_train.shape[1], C))
S_est = np.zeros((DOA_train.shape[1], L, 2))
S_abs = np.zeros((DOA_train.shape[1], 2 * L))
Signal_wave = np.zeros((DOA_train.shape[1], 2, snapshot)) + 1j * np.zeros((DOA_train.shape[1], 2, snapshot))
Rx = np.zeros((DOA_train.shape[1], M, M)) + 1j * np.zeros((DOA_train.shape[1], M, M))
Rx_flatten = np.zeros((DOA_train.shape[1], M ** 2, 1)) + 1j * np.zeros((DOA_train.shape[1], M ** 2, 1))
X = np.zeros((DOA_train.shape[1], M, snapshot)) + 1j * np.zeros((DOA_train.shape[1], M, snapshot))

for i in range(DOA_train.shape[1]):
    X1, _, s1 = signal_generate(M, snapshot, DOA_train[0, i])
    X2, _, s2 = signal_generate(M, snapshot, DOA_train[1, i])
    sPadding = np.zeros((L, snapshot)) + 1j * np.zeros((L, snapshot))
    sPadding[int(np.round(DOA_train[0, i])) - D_start, :] = s1
    sPadding[int(np.round(DOA_train[1, i])) - D_start, :] = s2
    SSH = np.matmul(sPadding, sPadding.conj().T) / snapshot
    X1_power = (np.matmul(s1, s1.conj().T)).real / snapshot
    X2_power = (np.matmul(s2, s2.conj().T)).real / snapshot
    SNR = np.random.uniform(-10, 10)
    temp1 = add_awgn(X1, X1_power, SNR)
    SNR = np.random.uniform(-10, 10)
    temp2 = add_awgn(X2, X2_power, SNR)
    X[i] = temp1 + temp2
    # print(X[i].shape)
    R_est[i, :], Rx[i, :, :] = feature_extract_R(X[i])
    Rx_flatten[i, :] = Rx[i, :, :].transpose().reshape(-1, 1)
    # Extract feature vector and match to DOA grid
    temp = np.matmul(H.conj().T, Rx_flatten[i, :]).reshape(-1)
    temp = temp / np.linalg.norm(temp, ord=2)
    S_est[i, :, 0] = np.real(temp)
    S_est[i, :, 1] = np.imag(temp)
    S_abs[i, :] = np.concatenate((np.real(temp), np.imag(temp)))
    S_label[i, int(np.round(DOA_train[0, i]) - D_start)] = 1
    S_label[i, int(np.round(DOA_train[1, i]) - D_start)] = 1
    Signal_wave[i, 0, :] = s1
    Signal_wave[i, 1, :] = s2
    Label_SSH[i] = np.diag(SSH).real

# i = 70
# plt.axvline(DOA_train[0, i])
# plt.axvline(DOA_train[1, i])
# plt.plot(theta, S_est[i, :, 0], label='Real[data]')
# plt.plot(theta, S_est[i, :, 1], label='Imag[data]')
# plt.plot(theta, S_label[i, :], label='Label')
# plt.plot(theta, Label_SSH[i, :], label="SSH")
# # plt.xlim([-60, 60])
# plt.grid(True)
# plt.legend()
# plt.show()

# Save the data as .h5 file
with h5py.File('../Data/TestData_var_N.h5', 'w') as f:
    f.create_dataset('Label', data=S_label)
    f.create_dataset('LabelPower', data=Label_SSH)
    f.create_dataset('Dictionary', data=H)
    f.create_dataset('CovarianceMatrix', data=Rx)
    f.create_dataset('DOA_train', data=DOA_train)
    f.create_dataset('CovarianceMatrixFlatten', data=Rx_flatten)
    f.create_dataset('RawData', data=X)
    f.create_dataset('PseudoSpectrum', data=S_est)
    f.create_dataset('PseudoSpectrumAbs', data=S_abs)
