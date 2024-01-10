import DoaMethods
from DoaMethods.configs import config_test_SNR as config
from DoaMethods.configs import name, is_checkpoint, ModelMethods, DataMethods, UnfoldingMethods
import matplotlib.pyplot as plt

TestCurve = DoaMethods.TestCurve(dir_test=config['data_path'])
if name in UnfoldingMethods or name in DataMethods:
    if is_checkpoint:
        predict, _ = TestCurve.test_model(name=name, model_dir=f"{config['model_path']}",
                                          num_layers=config['num_layers'], device=config['device'])
    else:
        predict, _ = TestCurve.test_model(name=name, model_dir=f"{config['model_path']}/model_30.pth",
                                          num_layers=config['num_layers'], device=config['device'])
    peak = TestCurve.find_peak(predict.detach().numpy())

elif name in ModelMethods:
    predict = TestCurve.test_alg(name=name)
    peak = TestCurve.find_peak(predict)
else:
    raise ValueError("Wrong name!")
_, RMSE, NMSE, prob = TestCurve.calculate_error(peak)

plt.style.use(['science', 'ieee', 'grid'])
snr_list = [i for i in range(-12, 13, 1)]

plt.plot(snr_list, RMSE, label=name)
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

plt.plot(snr_list, NMSE, label=name)
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.title("NMSE vs SNR")
plt.ylim(-35, -5)
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/varSNR_NMSE.pdf")
plt.show()
plt.close()

with open(f"{config['result_path']}/varSNR{config['testSNR_interval']}.csv", 'w') as f:
    f.write("SNR, RMSE, NMSE, prob\n")
    for i in range(len(snr_list)):
        f.write(f"{snr_list[i]}, {RMSE[i]}, {NMSE[i]}, {prob[i]}\n")



