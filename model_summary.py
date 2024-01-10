from torchsummary import summary
from DoaMethods.UnfoldingMethods import AMI_LISTA, LISTA, CPSS_LISTA
from DoaMethods.DataMethods import DCNN
import torch

if __name__ == '__main__':
    # model = AMI_LISTA(torch.rand(64, 121))
    # summary(model, (64, 1))

    model = LISTA(torch.rand(64, 121))
    summary(model, (64, 1))

    # model = CPSS_LISTA(10, 2)
    # summary(model, (64, 1))

    # model = DCNN()
    # summary(model, (2, 121))
