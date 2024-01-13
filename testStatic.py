import DoaMethods
from configs import config_test_static as config
from configs import name, is_checkpoint, ModelMethods, DataMethods, UnfoldingMethods
import matplotlib.pyplot as plt
mode = config['mode']

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
if mode == 'SNR':
    x_tricks = [i for i in range(-12, 13, 1)]
elif mode == 'Separation':
    x_tricks = [i for i in range(2, 21, 1)]
elif mode == 'Snapshots':
    x_tricks = [i for i in range(10, 410, 10)]

plt.plot(x_tricks, RMSE, label=name)
if mode == 'SNR':
    plt.xlabel('SNR/dB')
elif mode == 'Separation':
    plt.xlabel('Angle Separation/$^{\circ}$')
elif mode == 'Snapshots':
    plt.xlabel('Snapshots')
else:
    raise ValueError("Wrong mode!")
# plt.ylim(1e-1, 30)
plt.ylabel('RMSE/$^{\circ}$')
plt.yscale('log')
plt.title(f"RMSE vs {mode}")
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/var{mode}_RMSE.pdf")
plt.show()
plt.close()

plt.plot(x_tricks, NMSE, label=name)
if mode == 'SNR':
    plt.xlabel('SNR/dB')
elif mode == 'Separation':
    plt.xlabel('Angle Separation/$^{\circ}$')
elif mode == 'Snapshots':
    plt.xlabel('Snapshots')
else:
    raise ValueError("Wrong mode!")
plt.ylabel('NMSE/dB')
plt.title(f"NMSE vs {mode}")
# plt.ylim(-45, -5)
plt.legend(loc='upper right', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f"{config['figure_path']}/var{mode}_NMSE.pdf")
plt.show()
plt.close()

with open(f"{config['result_path']}/var{mode}{config['testSNR_interval']}.csv", 'w') as f:
    f.write("SNR, RMSE, NMSE, prob\n")
    for i in range(len(x_tricks)):
        f.write(f"{x_tricks[i]}, {RMSE[i]}, {NMSE[i]}, {prob[i]}\n")



