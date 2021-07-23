import torch
import torch.nn as nn

import time
from datetime import timedelta

from utils import get_network, get_dataloader

experiments = {
    'networks': ['coconv4_alexnet', 'coconv4_resnet18', 'coconv4_mobilenetv2'],
    'datasets': ['cifar100', 'tiny']
}

for i in range(3):
    for j in range(2):
        net = get_network(experiments['networks'][i], experiments['datasets'][j], 'cpu', 'sigmoid')
        x = torch.randn(128, 3, 32, 32) if experiments['datasets'][j] == 'cifar100' else torch.randn(128, 3, 64, 64)
        start = time.time()
        for _ in range(10):
            net(x)
        end = time.time()
        print('{} - {}: {} ({} s)'.format(
            experiments['networks'][i], 
            experiments['datasets'][j], 
            str(timedelta(seconds=int(end - start)) ),
            end - start
            )
        )