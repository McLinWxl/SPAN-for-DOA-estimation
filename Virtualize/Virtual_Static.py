import numpy as np
import matplotlib.pyplot as plt
from configs import testSNR_interval, config_test_static, num_sensors
mode = config_test_static['mode']
import numpy as np

is_ablition = True


def load_data(names, mode, testSNR_interval):
    data = {}
    for name in names:
        if mode in ['SNR', 'Snapshots']:
            varSNR_dir = f"../Result_mini/{name}/var{mode}{testSNR_interval}_{num_sensors}.csv"
        elif mode == 'Separation':
            varSNR_dir = f"../Result_mini/{name}/var{mode}_{num_sensors}.csv"
        elif mode == 'Sensors':
             varSNR_dir = f"../Result_mini/{name}/var{mode}.csv"
        varSNR, RMSE, NMSE, prob = np.loadtxt(varSNR_dir, delimiter=',', skiprows=1).T
        data[name] = {'varSNR': varSNR, 'RMSE': RMSE, 'NMSE': NMSE, 'prob': prob}
    return data

def plot_graph(data, plot_config, names, mode, ylabel, save_as):
    for name in names:
        if name in plot_config.keys():
            plt.plot(data[name]['varSNR'], data[name][ylabel], **plot_config[name])
    plt.xlabel('SNR(dB)' if mode == 'SNR' else 'Angle Separation()$^{\circ}$)' if mode == 'Separation' else 'Snapshots' if mode == 'Snapshots' else ValueError("Wrong mode!"))
    plt.ylabel(f'{ylabel}($^\circ$)')
    if ylabel == 'RMSE':
        plt.ylim(1e-1, 30)
    elif ylabel == 'NMSE':
        plt.ylim(-45, -5)
    else:
        plt.ylim(0, 1)
    plt.yscale('log') if ylabel == 'RMSE' else None
    plt.title(f"{ylabel} vs {mode}" if ylabel != 'prob' else "Accuracy of Estimation")
    plt.legend(loc='upper right' if ylabel != 'prob' else 'lower right', prop={'size': 5})
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
    plt.savefig(f"../Figure_mini/Static/Ab_var{mode}{testSNR_interval}_{save_as}.pdf" if is_ablition else f"../Figure_mini/Static/var{mode}{testSNR_interval}_{save_as}.pdf")
    plt.show()
    plt.close()

if is_ablition:
    names = ["ALISTA-SS-LF12", "ALISTA-SS-LF24", "ALISTA-SS-LF48"]  #
    plot_config = {
        "ALISTA-10": {"label": 'ALISTA'},
        "ALISTA-20": {"label": 'ALISTA-20'},
        "ALISTA-SS-LF12": {"label": 'ALISTA-SS-12'},
        "ALISTA-SS-LF24": {"label": 'ALISTA-SS-24'},
        "ALISTA-SS-LF48": {"label": 'ALISTA-SS-48'},
        "ALISTA-SS-LF96": {"label": 'ALISTA-SS-96'},
    }
    data = load_data(names, mode, testSNR_interval)
    plt.style.use(['science', 'ieee', 'grid'])
    plot_graph(data, plot_config, names, mode, 'RMSE', 'RMSE')
    plot_graph(data, plot_config, names, mode, 'NMSE', 'NMSE')
    plot_graph(data, plot_config, names, mode, 'prob', 'accuracy')
else:
    #     names = ["AMI-LF10", "LISTA-10", "MUSIC", "MVDR", "DCNN", "ALISTA-10"]
    names = ["MUSIC", "DCNN", "MVDR"] #
    plot_config = {
        "AMI-LF10": {"label": 'AMI-LISTA'},
        "LISTA-10": {"label": 'LISTA'},
        "CPSS-10": {"label": 'CPSS-LISTA'},
        "MUSIC": {"label": 'MUSIC'},
        "MVDR": {"label": 'MVDR'},
        "DCNN": {"label": 'DCNN'},
        "ALISTA-SS-LF80": {"label": 'ALISTA-SS-80'},
    }
    data = load_data(names, mode, testSNR_interval)
    plt.style.use(['science', 'ieee', 'grid'])
    plot_graph(data, plot_config, names, mode, 'RMSE', 'RMSE')
    plot_graph(data, plot_config, names, mode, 'NMSE', 'NMSE')
    plot_graph(data, plot_config, names, mode, 'prob', 'accuracy')