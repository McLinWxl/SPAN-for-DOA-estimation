import itertools
import numpy as np
import h5py
from DataGenerator import feature_extract_R, add_awgn  # type: ignore
from rich.progress import track


def signal_generate_positionErrored(M, snapshot, DOA, errorRate = 0, x_root=0.5, y_root=0.5):
    a_Perf = np.exp(1j * np.pi * np.arange(M)[:, np.newaxis] * np.sin(np.deg2rad(DOA)))
    a_Imperf = np.exp(1j * np.pi * ((x_root * np.sign(np.random.randn(M, 1)) * np.ones((M, 1)) * errorRate) * np.sin(np.deg2rad(DOA)) +
                                    (y_root * np.sign(np.random.randn(M, 1)) * np.ones((M, 1)) * errorRate) * np.cos(np.deg2rad(DOA))))
    a_Imperf[0] = 1 + 0j
    a = a_Imperf * a_Perf
    s = (np.random.randn(1, snapshot) + 1j * np.random.randn(1, snapshot))
    return np.matmul(a, s),a,s

times = 1000

errorRate = 0
M = 8
snapshot = 256
C = M * (M - 1)
error_list = np.arange(0, 1.01, 0.05)
SNR = 0

D_start = -60
D_stop = 60

DOA_test = np.zeros((len(error_list), times, 2))
for idx, k in itertools.product(range(len(error_list)), range(times)):
    DOA_test[idx, k, 0] = -5 + 1 * np.random.uniform(-1, 1)
    DOA_test[idx, k, 1] = 10 + 1 * np.random.uniform(-1, 1)

theta = np.arange(D_start, D_stop + 1).reshape(-1)
L = len(theta)
S_label = np.zeros((DOA_test.shape[0], DOA_test.shape[1], L))
Label_power = np.zeros((DOA_test.shape[0], DOA_test.shape[1], L))
R_est = np.zeros((DOA_test.shape[0], DOA_test.shape[1], C))
S_est = np.zeros((DOA_test.shape[0], DOA_test.shape[1], L, 2))
S_abs = np.zeros((DOA_test.shape[0], DOA_test.shape[1], 2 * L))
Signal_wave = np.zeros((DOA_test.shape[0], DOA_test.shape[1], 2, snapshot)) + 1j * np.zeros(
    (DOA_test.shape[0], DOA_test.shape[1], 2, snapshot))
Rx = np.zeros((DOA_test.shape[0], DOA_test.shape[1], M, M)) + 1j * np.zeros(
    (DOA_test.shape[0], DOA_test.shape[1], M, M))
Rx_flatten = np.zeros((DOA_test.shape[0], DOA_test.shape[1], M ** 2, 1)) + 1j * np.zeros(
    (DOA_test.shape[0], DOA_test.shape[1], M ** 2, 1))
X = np.zeros((DOA_test.shape[0], DOA_test.shape[1], M, snapshot)) + 1j * np.zeros(
    (DOA_test.shape[0], DOA_test.shape[1], M, snapshot))

A = np.exp(1j * np.pi * np.arange(M)[:, np.newaxis] * np.sin(np.deg2rad(theta)))
H = np.zeros((M ** 2, L), dtype=np.complex128)
for i in range(M):
    s = np.exp(-1j * np.pi * (i) * np.sin(np.deg2rad(theta)))
    B = np.diag(s)
    fhi = np.matmul(A, B)
    H[i * M:(i + 1) * M, :] = fhi

for idx in track(range(len(error_list)), description="Generating data"):
    for i in range(DOA_test.shape[1]):
        X1, _, s1 = signal_generate_positionErrored(M, snapshot, DOA_test[idx, i, 0], errorRate=error_list[idx],
                                                    x_root=0.5, y_root=0.5)
        X2, _, s2 = signal_generate_positionErrored(M, snapshot, DOA_test[idx, i, 1], errorRate=error_list[idx],
                                                    x_root=0.5, y_root=0.5)
        X1_power = (np.matmul(s1, s1.conj().T)).real / snapshot
        X2_power = (np.matmul(s2, s2.conj().T)).real / snapshot
        temp1 = add_awgn(X1, X1_power, SNR)
        temp2 = add_awgn(X2, X2_power, SNR)
        X[idx, i] = temp1 + temp2
        R_est[idx, i, :], Rx[idx, i, :, :] = feature_extract_R(X[idx, i])
        Rx_flatten[idx, i] = Rx[idx, i].transpose().reshape(-1, 1)
        # Extract feature vector and match to DOA grid
        temp = np.matmul(H.conj().T, Rx_flatten[idx, i]).reshape(-1)
        temp = temp / np.linalg.norm(temp, ord=2)
        S_est[idx, i, :, 0] = np.real(temp)
        S_est[idx, i, :, 1] = np.imag(temp)
        S_abs[idx, i, :] = np.concatenate((np.real(temp), np.imag(temp)))
        S_label[idx, i, int(np.round(DOA_test[idx, i, 0]) - D_start)] = 1
        S_label[idx, i, int(np.round(DOA_test[idx, i, 1]) - D_start)] = 1
        Label_power[idx, i, int(np.round(DOA_test[idx, i, 0]) - D_start)] = X1_power
        Label_power[idx, i, int(np.round(DOA_test[idx, i, 1]) - D_start)] = X2_power
        Signal_wave[idx, i, 0, :] = s1
        Signal_wave[idx, i, 1, :] = s2

print(Rx_flatten.shape)

# Save the data as .mat file
with h5py.File('../Data/TestData_Imp.h5', 'w') as f:
    f.create_dataset('Label', data=S_label)
    f.create_dataset('LabelPower', data=Label_power)
    f.create_dataset('Dictionary', data=H)
    f.create_dataset('CovarianceMatrix', data=Rx)
    f.create_dataset('DOA_train', data=DOA_test)
    f.create_dataset('CovarianceMatrixFlatten', data=Rx_flatten)
    f.create_dataset('RawData', data=X)
    f.create_dataset('PseudoSpectrum', data=S_est)
    f.create_dataset('PseudoSpectrumAbs', data=S_abs)
