import h5py
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track


def signal_generate(num_sensors, num_snapshot, DOA):
    steering_vector = np.exp(1j * np.pi * np.arange(num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(DOA)))
    signal_wave = (np.random.randn(1, num_snapshot) + 1j * np.random.randn(1, num_snapshot))
    return np.matmul(steering_vector, signal_wave), steering_vector, signal_wave


def feature_extract_R(X):
    """
    Calculate the covariance matrix of the input signal
    """
    num_sensors, num_snapshots = X.shape
    covariance_matrix = np.matmul(X, X.conj().T) / num_snapshots  # calculate the array covariance matrix
    r_doa = np.zeros((num_sensors * (num_sensors - 1)) // 2) + 1j * np.zeros((num_sensors * (num_sensors - 1)) // 2)
    k = 0
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            r_doa[k] = covariance_matrix[i, j]
            k += 1

    r_doa = r_doa / np.linalg.norm(r_doa)
    r_doa1 = np.concatenate((np.real(r_doa), np.imag(r_doa)))  # concatenate the real and imaginary parts
    return r_doa1.reshape(1, -1), covariance_matrix


def add_awgn(signal, power, snr_db):
    # print(snr_db)
    num_sensors, num_snapshot = signal.shape
    noise_power = power / (10 ** (snr_db / 10))
    noise2add = np.sqrt(noise_power) * (
            np.random.normal(0, np.sqrt(2) / 2, (num_sensors, num_snapshot))
            + 1j * np.random.normal(0, np.sqrt(2) / 2, (num_sensors, num_snapshot)))
    return signal + noise2add


if __name__ == '__main__':
    M = 8
    snapshot = 256
    f0 = 1e6
    fc = 1e6
    lam = fc / f0
    fs = 4 * f0
    C = M * (M - 1)
    SNR = -10
    noise = 1

    DOA11 = []
    DOA22 = []
    k1 = np.arange(1, 41, 1)
    k = np.tile(k1, 30)
    D_start = -60
    D_stop = 60

    for l in range(len(k)):
        DOA1 = np.arange(D_start, D_stop - k[l] + 1)
        DOA2 = np.arange(D_start + k[l], D_stop + 1)
        DOA11 = np.concatenate([DOA11, DOA1])
        DOA22 = np.concatenate([DOA22, DOA2])
    DOA_train = np.vstack([DOA11, DOA22])

    theta = np.arange(D_start, D_stop + 1).reshape(-1)
    L = len(theta)

    A = np.exp(1j * np.pi * np.arange(M)[:, np.newaxis] * np.sin(np.deg2rad(theta)))

    H = np.zeros((M ** 2, L), dtype=np.complex128)

    for i in range(M):
        s = np.exp(-1j * np.pi * (i) * np.sin(np.deg2rad(theta)))
        B = np.diag(s)
        fhi = np.matmul(A, B)
        H[i * M:(i + 1) * M, :] = fhi

    S_label = np.zeros((DOA_train.shape[1], L))
    R_est = np.zeros((DOA_train.shape[1], C))
    S_est = np.zeros((DOA_train.shape[1], L, 2))
    S_abs = np.zeros((DOA_train.shape[1], 2 * L))
    Label_SSH = np.zeros((DOA_train.shape[1], L))
    Signal_wave = np.zeros((DOA_train.shape[1], 2, snapshot)) + 1j * np.zeros((DOA_train.shape[1], 2, snapshot))
    Rx = np.zeros((DOA_train.shape[1], M, M)) + 1j * np.zeros((DOA_train.shape[1], M, M))
    Rx_flatten = np.zeros((DOA_train.shape[1], M ** 2, 1)) + 1j * np.zeros((DOA_train.shape[1], M ** 2, 1))
    X = np.zeros((DOA_train.shape[1], M, snapshot)) + 1j * np.zeros((DOA_train.shape[1], M, snapshot))
    # Generate signals and estimate DOA
    for i in track(range(DOA_train.shape[1])):
        X1, a1, s1 = signal_generate(M, snapshot, DOA_train[0, i])
        X2, a2, s2 = signal_generate(M, snapshot, DOA_train[1, i])
        sPadding = np.zeros((L, snapshot)) + 1j * np.zeros((L, snapshot))
        sPadding[int(np.round(DOA_train[0, i])) - D_start, :] = s1
        sPadding[int(np.round(DOA_train[1, i])) - D_start, :] = s2
        SSH = np.matmul(sPadding, sPadding.conj().T) / snapshot

        # The power below is power ** 2
        X1_power = (np.matmul(s1, s1.conj().T)).real / snapshot
        X2_power = (np.matmul(s2, s2.conj().T)).real / snapshot

        temp1 = add_awgn(X1, X1_power, SNR * np.random.uniform(0, 1))
        temp2 = add_awgn(X2, X2_power, SNR * np.random.uniform(0, 1))

        X[i] = temp1 + temp2
        # Compute covariance matrix
        R_est[i, :], Rx[i, :, :] = feature_extract_R(X[i])
        Rx_flatten[i, :] = Rx[i, :, :].transpose().reshape(-1, 1)
        # Rx_flatten[i,:] = Rx_flatten[i,:] / np.linalg.norm(Rx_flatten[i,:], ord=2)
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
    # Plot the estimated DOA values for the 7000th iteration

    print(Rx_flatten.shape)

    i = 7000
    plt.plot(theta, S_est[i, :, 0], label='Real[data]')
    plt.plot(theta, S_est[i, :, 1], label='Imag[data]')
    plt.plot(theta, S_label[i, :], label='Label')
    plt.xlim([-60, 60])
    plt.grid(True)
    plt.legend()
    plt.show()

    with h5py.File('../Data/TrainData.h5', 'w') as f:
        f.create_dataset('Label', data=S_label)
        f.create_dataset('LabelPower', data=Label_SSH)
        f.create_dataset('Dictionary', data=H)
        f.create_dataset('CovarianceMatrix', data=Rx)
        f.create_dataset('DOA_train', data=DOA_train)
        f.create_dataset('CovarianceMatrixFlatten', data=Rx_flatten)
        f.create_dataset('RawData', data=X)
        f.create_dataset('PseudoSpectrum', data=S_est)
        f.create_dataset('PseudoSpectrumAbs', data=S_abs)
