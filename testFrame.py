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

model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device']).load_model(f"{config['model_path']}/model_{epoch_read}.pth")

model.eval()
mse_val_last = 0
with torch.no_grad():
    for covariance_array, label in loader:
        label = label.to(config['device'])
        covariance_array = covariance_array.to(config['device'])
        label = label.to(config['device'])
        label /= torch.norm(label, dim=1, keepdim=True)
        label /= torch.sqrt(torch.tensor(2))
        loss_value = 0
        output, layers_output_val = model(covariance_array)

idx = 1
plt.plot(output[idx].detach().numpy())
for k in range(label.shape[1]):
    if label[idx, k] != 0:
        plt.axvline(x=k, color='r')
plt.show()
