import numpy as np
import matplotlib.pyplot as plt

names = ["AMI-LF10", "LISTA-10", "CPSS-10", "MUSIC", "MVDR", "DCNN"]
for name in names:
    varSNR_dir = f"../Result/{name}/varSNR16.csv"
    varSNR = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 0]
    if name == "AMI-LF10":
        RMSE_Ami = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_Ami = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_Ami = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "LISTA-10":
        RMSE_Lista = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_Lista = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_Lista = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "CPSS-10":
        RMSE_Cpss = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_Cpss = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_Cpss = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "MUSIC":
        RMSE_Music = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_Music = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_Music = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "MVDR":
        RMSE_Mvdr = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_Mvdr = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_Mvdr = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    elif name == "DCNN":
        RMSE_Dcnn = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 1]
        NMSE_Dcnn = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 2]
        prob_Dcnn = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1)[:, 3]
    else:
        raise ValueError("Wrong name!")

plt.style.use(['science', 'ieee', 'grid'])
plt.plot(varSNR, RMSE_Ami, label='AMI-LISTA')
plt.plot(varSNR, RMSE_Lista, label='LISTA')
plt.plot(varSNR, RMSE_Cpss, label='CPSS-LISTA')
plt.plot(varSNR, RMSE_Music, label='MUSIC')
plt.plot(varSNR, RMSE_Mvdr, label='MVDR', color='orange')
plt.plot(varSNR, RMSE_Dcnn, label='DCNN', color='fuchsia')
plt.xlabel('SNR/dB')
plt.ylabel('RMSE/$^{\circ}$')
plt.ylim(1e-1, 30)
plt.yscale('log')
plt.title("RMSE vs SNR")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"../Figure/Static/varSNR16_RMSE.pdf")
plt.show()


