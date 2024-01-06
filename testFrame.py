import DoaMethods
import os
import torch.utils.data
from configs import config_test as config
from configs import name
import matplotlib.pyplot as plt

epoch_read = config['epoch']

dataset = DoaMethods.MakeDataset(config['data_path'])
print(len(dataset))

loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

dictionary = torch.from_numpy(dataset.get_dictionary())

model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device']).load_model(f"{config['model_path']}/model_150.pth")

model.eval()
mse_val_last = 0
with torch.no_grad():
    for covariance_array, label in loader:
        label = label.to(config['device'])
        covariance_array = covariance_array.to(config['device'])
        output, layers_output_val = model(covariance_array)

idx = 1
for i in range(config['num_layers']):
    plt.plot(layers_output_val[idx, i].detach().numpy())
    for k in range(label.shape[1]):
        if label[idx, k] != 0:
            plt.scatter(k, label[idx, k], c='r')
    plt.savefig(f"{config['figure_path']}/layer_{i}.pdf")
    plt.show()
    plt.close()
