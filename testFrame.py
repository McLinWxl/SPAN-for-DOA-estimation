import numpy as np

import DoaMethods
import torch.utils.data
from DoaMethods.configs import config_test as config
from DoaMethods.configs import name, DataMethods, UnfoldingMethods, ModelMethods
import matplotlib.pyplot as plt
import numpy

epoch_read = config['epoch']

dataset = DoaMethods.MakeDataset(config['data_path'])
print(len(dataset))

loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

dictionary_numpy = dataset.get_dictionary()
dictionary = torch.from_numpy(dataset.get_dictionary())

if name in UnfoldingMethods or name in DataMethods:
    model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device']).load_model(f"{config['model_path']}/best.pth")

    model.eval()
    mse_val_last = 0
    with torch.no_grad():
        for data, label in loader:
            label = label.to(config['device'])
            data = data.to(config['device'])
            if name in UnfoldingMethods:
                output, layers_output_val = model(data)
            elif name in DataMethods:
                output = model(data)
            else:
                raise ValueError("Wrong name!")

elif name in ModelMethods:
    algorithm = DoaMethods.ModelMethods.ModelMethods(dictionary=dictionary_numpy)
    predict = np.zeros((len(dataset), dataset.num_meshes, 1))
    if name == 'ISTA':
        covariance_vector = dataset.covariance_vector
        label = dataset.label
        for i in range(covariance_vector.shape[0]):
            predict[i] = algorithm.ISTA(covariance_vector[i])

    elif name == 'MUSIC':
        covariance_matrix = dataset.covariance_matrix_clean
        label = dataset.label
        for i in range(covariance_matrix.shape[0]):
            predict[i] = algorithm.MUSIC(covariance_matrix[i])

    elif name == 'SBL':
        raw_data = dataset.dataset_h5['RawData'][()]
        label = dataset.label
        for i in range(raw_data.shape[0]):
            predict[i] = algorithm.SBL(raw_data[i]).reshape(-1, 1)

    elif name == 'MVDR':
        covariance_matrix = dataset.covariance_matrix_clean
        label = dataset.label
        for i in range(covariance_matrix.shape[0]):
            predict[i] = algorithm.MVDR(covariance_matrix[i])

    else:
        raise ValueError("Wrong name!")


idxs = [0, 1, 2, 3, 4]
for idx in idxs:
    if name in UnfoldingMethods:
        plt.rcParams['font.family'] = 'Times New Roman'
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x_idx = numpy.arange(121) - 60
        for i in range(config['num_layers']):
            to_plot = layers_output_val[idx, i].detach().numpy().reshape(-1)
            ax.plot(x_idx, numpy.ones(121) * (config['num_layers'] - i), to_plot)
        ax.set_box_aspect((20, 60, 13))
        ax.view_init(27, -28)
        # no meshed and no grid and no axis
        # ax.set_axis_off()
        # ax.grid(False)
        ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_zticks([])
        # ax.set_xlabel('Angle $^{\circ}$')
        # ax.set_ylabel('Layer')
        # ax.set_zlabel('Amp.')
        # Delete the white space around the figure
        plt.subplots_adjust(left=0, right=0.9, bottom=0, top=1)
        plt.savefig(f"{config['figure_path']}/layer_{idx}.pdf")
        plt.show()
        plt.close()

        plt.style.use(['science', 'ieee', 'grid'])
        plt.plot(output[idx].detach().numpy(), label=name)
        for i in range(output.shape[1]):
            if label[idx, i]:
                plt.axvline(x=i, color='red', linestyle='--')
        plt.xlabel('Angle $^{\circ}$')
        plt.ylabel('Amp.')
        plt.legend()
        plt.savefig(f"{config['figure_path']}/output_{idx}.pdf")
        plt.show()

    elif name in DataMethods:
        plt.style.use(['science', 'ieee', 'grid'])
        # plt.rcParams['figure.dpi'] = 1000
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.plot(output[idx].detach().numpy(), label=name)
        for i in range(output.shape[1]):
            if label[idx, i]:
                plt.axvline(x=i, color='red', linestyle='--')
        plt.xlabel('Angle $^{\circ}$')
        plt.ylabel('Amp.')
        plt.legend()
        plt.savefig(f"{config['figure_path']}/output_{idx}.pdf")
        plt.show()

    elif name in ModelMethods:
        plt.style.use(['science', 'ieee', 'grid'])
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.plot(predict[idx].reshape(-1), label=name)
        for i in range(predict.shape[1]):
            if label[idx, i]:
                plt.axvline(x=i, color='red', linestyle='--')
        plt.xlabel('Angle $^{\circ}$')
        plt.ylabel('Amp.')
        plt.legend()
        plt.savefig(f"{config['figure_path']}/output_{idx}.pdf")
        plt.show()

