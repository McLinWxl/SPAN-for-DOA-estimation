import DoaMethods
from configs import config_test_SNR as config
from configs import name
import matplotlib.pyplot as plt

TestCurve = DoaMethods.TestCurve(dir_test=config['data_path'])
AMI_predict, _ = TestCurve.test_model(name=name, model_dir=f"{config['model_path']}/best.pth",
                                      num_layers=config['num_layers'], device=config['device'])
AMI_peak = TestCurve.find_peak(AMI_predict.detach().numpy())
_, AMI_RMSE, AMI_NMSE, AMI_prob = TestCurve.calculate_error(AMI_peak)

plt.style.use(['science', 'ieee', 'grid'])
snr_list = [i for i in range(-12, 13, 1)]

plt.plot(snr_list, AMI_RMSE, label='AMI')
plt.xlabel('SNR/dB')
plt.ylim(1e-1, 30)
plt.ylabel('RMSE/$^{\circ}$')
plt.yscale('log')
plt.title("RMSE vs SNR")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/varSNR_RMSE.pdf")
plt.show()
plt.close()

plt.plot(snr_list, AMI_NMSE, label='AMI')
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.title("NMSE vs SNR")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/varSNR_NMSE.pdf")
plt.show()
plt.close()

with open(f"{config['result_path']}/varSNR.csv", 'w') as f:
    f.write("SNR, RMSE, NMSE, prob\n")
    for i in range(len(snr_list)):
        f.write(f"{snr_list[i]}, {AMI_RMSE[i]}, {AMI_NMSE[i]}, {AMI_prob[i]}\n")



