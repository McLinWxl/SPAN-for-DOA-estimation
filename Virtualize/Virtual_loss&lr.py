import numpy as np
import matplotlib.pyplot as plt
from DoaMethods.configs import config_test_SNR as config

lr_dir = f"{config['result_path']}/lr.csv"
loss_dir = f"{config['result_path']}/loss.csv"
varSNR_dir = f"{config['result_path']}/varSNR{config['testSNR_interval']}.csv"

# lr_dir = '/Volumes/WangXinLin/GitLibrary/UnfoldingDOA/Result/AMI-LF10_old/lr.csv'
# loss_dir = '/Volumes/WangXinLin/GitLibrary/UnfoldingDOA/Result/AMI-LF10_old/loss.csv'

is_lr = True
is_SNR = False
start = 0
if is_lr:
    epoch = np.loadtxt(lr_dir, delimiter=',', skiprows=1)[start:, 0]
    lr = np.loadtxt(lr_dir, delimiter=',', skiprows=1)[start:, 1] - start
    loss_train = np.loadtxt(loss_dir, delimiter=',', skiprows=1)[start:, 1]
    loss_valid = np.loadtxt(loss_dir, delimiter=',', skiprows=1)[start:, 2]
if is_SNR:
    varSNR = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[start:, 0]
    RMSE = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[start:, 1]
    NMSE = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[start:, 2]
    prob = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[start:, 3]

if is_lr:
    plt.style.use(['science', 'ieee', 'grid'])
    plt.plot(epoch, loss_train, label='Train')
    plt.plot(epoch, loss_valid, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss vs Epoch")
    plt.legend(loc='upper right', prop={'size': 5})
    plt.show()

    plt.close()
    plt.plot(epoch, lr)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title("Learning Rate vs Epoch")
    plt.show()

if is_SNR:
    plt.close()
    plt.plot(varSNR, RMSE)
    plt.xlabel('SNR/dB')
    plt.ylabel('RMSE/$^{\circ}$')
    plt.ylim(1e-1, 30)
    plt.yscale('log')
    plt.title("RMSE vs SNR")
    plt.show()

    plt.close()
    plt.plot(varSNR, NMSE)
    plt.xlabel('SNR/dB')
    plt.ylim(-35, -5)
    plt.ylabel('RMSE/dB')
    plt.title("NMSE vs SNR")
    plt.show()

    plt.close()
    plt.plot(varSNR, prob)
    plt.xlabel('SNR/dB')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs SNR')
    plt.show()

pass