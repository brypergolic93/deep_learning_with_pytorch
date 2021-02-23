# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_printoptions(edgeitems = 2)
torch.manual_seed(123)


from torchvision import models

alexnet = models.AlexNet()

resnet = models.resnet101(pretrained = True)

print(resnet)