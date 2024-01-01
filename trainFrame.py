import DoaMethods
import torch.utils.data
from configs import config, name

dataset = DoaMethods.MakeDataset(config['data_path'])
print(len(dataset))
train_set, valid_set = torch.utils.data.random_split(dataset,
                                                     [int(len(dataset) * 0.8),
                                                      len(dataset) - int(len(dataset) * 0.8)])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False)

dictionary = torch.from_numpy(dataset.get_dictionary())

model = DoaMethods.functions.ReadModel(name=name, dictionary=dictionary, num_layers=config['num_layers'], device=config['device']).get_model()
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

loss_best = 100
for epoch in range(config['epoch']+1):
    if epoch < 15 or epoch % 10 == 0:
        torch.save({'model': model.state_dict()}, f"{config['model_path']}/model_{epoch}.pth")
    model.train()
    mse_train_last = 0
    for covariance_vector, label in train_loader:
        mse_loss = 0
        label = label.to(config['device'])
        covariance_vector = covariance_vector.to(config['device'])
        label = label.to(config['device'])
        label /= torch.norm(label, dim=1, keepdim=True)
        label /= torch.sqrt(torch.tensor(2))
        output, layers_output = model(covariance_vector)
        if config['LF']:
            for i in range(config['num_layers']):
                mse_loss = mse_loss + (loss(layers_output[:, i].to(torch.float32), label.to(torch.float32))) * (
                        torch.log(torch.tensor(i + 1)) + 1)
        else:
            mse_loss = loss(output.to(torch.float32), label.to(torch.float32))
        mse_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mse_train_last += mse_loss.item()
    mse_train_last /= len(train_loader)

    model.eval()
    mse_val_last = 0
    with torch.no_grad():
        for covariance_array, label in val_loader:
            label = label.to(config['device'])
            covariance_array = covariance_array.to(config['device'])
            label = label.to(config['device'])
            label /= torch.norm(label, dim=1, keepdim=True)
            label /= torch.sqrt(torch.tensor(2))
            loss_value = 0
            output, layers_output_val = model(covariance_array)
            if config['LF']:
                for i in range(config['num_layers']):
                    loss_value = loss_value + (
                        loss(layers_output_val[:, i].to(torch.float32), label.to(torch.float32))) * (
                                         torch.log(torch.tensor(i + 1)) + 1)
            else:
                loss_value = loss(output.to(torch.float32), label.to(torch.float32))
            mse_val_last += loss_value.item()
    mse_val_last /= len(val_loader)
    print(f"Epoch: {epoch}, Train Loss: {mse_train_last}, Valid Loss: {mse_val_last}")

    if mse_val_last < loss_best:
        loss_best = mse_val_last
        torch.save({'model': model.state_dict()}, f"{config['model_path']}/best.pth")
        print("Best model saved")

    # Save loss to .csv

    with open(f"{config['result_path']}/loss.csv", 'a+') as f:
        f.write(f"{epoch},{mse_train_last},{mse_val_last}\n")



