'''
    Used to train models on CIFAR-100 and Tiny ImageNet
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import yaml
import argparse
import time
from datetime import timedelta

from utils import calculate_acc, get_network, get_dataloader, init_params, count_parameters

parser = argparse.ArgumentParser(description='Training CoConv models')
parser.add_argument('--config', '-c', type=str, help='path of config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.full_load(f)

network_name = cfg['model']['type'] + str(cfg['model']['num_experts']) + '_' + cfg['model']['name']
# Files to record stuff to
RESULT_FILE = 'results.txt'
LOG_FILE = 'logs/{}-{}-b{}-e{}.txt'.format(network_name, 
                                cfg['misc']['dataset'], 
                                cfg['hyperparameters']['batch'], 
                                cfg['hyperparameters']['epochs'])

# Validation set length
VAL_LEN = 10000

# Dict to keep the final result
stats = {
    'best_acc': 0.0,
    'best_epoch': 0
}

# Device
device = torch.device('cuda' if (torch.cuda.is_available() and cfg['misc']['cuda']) else 'cpu')

# Dataloader
trainloader, testloader = get_dataloader(dataset=cfg['misc']['dataset'], batch_size=cfg['hyperparameters']['batch'])

# Get network
net = get_network(network=network_name, 
                dataset=cfg['misc']['dataset'], 
                device=device,
                activation=cfg['model']['routing_activation'])

# Handle multi-gpu
if cfg['misc']['cuda'] and cfg['misc']['ngpu'] > 1:
    net = nn.DataParallel(net, list(range(cfg['misc']['ngpu'])))

# Init parameters
init_params(net=net)

print('Training {} with {} parameters...'.format(network_name, count_parameters(net)))

net.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 
                            lr=cfg['hyperparameters']['learning_rate'], 
                            momentum=cfg['hyperparameters']['momentum'], 
                            weight_decay=cfg['hyperparameters']['weight_decay'])

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                        step_size=cfg['hyperparameters']['step_size'], 
                                        gamma=cfg['hyperparameters']['gamma'])

if cfg['track']['save'] and not cfg['track']['resume']:
    # Log basic hyper-params to log file
    with open(LOG_FILE, 'w') as f:
        f.write('Training model {}\n'.format(network_name))
        f.write('Hyper-parameters:\n')
        f.write('Epoch {}; Batch {}; LR {}; SGD Momentum {}; SGD Weight Decay {};\n'.format(
                str(cfg['hyperparameters']['epochs']), 
                str(cfg['hyperparameters']['batch']), 
                str(cfg['hyperparameters']['learning_rate']), 
                str(cfg['hyperparameters']['momentum']), 
                str(cfg['hyperparameters']['weight_decay'])
                )
        )
        f.write('LR Scheduler Step {}; LR Scheduler Gamma {}; {};\n'.format(
                str(cfg['hyperparameters']['step_size']), 
                str(cfg['hyperparameters']['gamma']), 
                str(cfg['misc']['dataset'])
                )
        )
        f.write('Epoch,TrainLoss,ValAcc\n')

if cfg['track']['resume']:
    checkpoint_path = cfg['track']['resume_path']
    state = torch.load(checkpoint_path)
    optimizer.load_state_dict(state['optimizer'])
    net.load_state_dict(state['net'])
    start_epoch = state['epoch']
    stats = state['stats'] if state['stats'] else { 'best_acc': 0.0, 'best_epoch': 0 }
else: # Train a new model from random initialization
    checkpoint_path = 'trained_nets/{}-{}-b{}-e{}.tar'.format(
                network_name, cfg['misc']['dataset'], 
                cfg['hyperparameters']['batch'], 
                cfg['hyperparameters']['epochs'])
    start_epoch = 0

# Train the model
start = time.time()
for epoch in range(start_epoch, cfg['hyperparameters']['epochs']):  # Iterate over the dataset multiple times

    training_loss = 0.0
    for i, data in enumerate(trainloader):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    # Calculate validation accuracy after every iteration
    net.eval()
    val_acc = calculate_acc(testloader, net, device)
    if val_acc > stats['best_acc']:
        stats['best_acc'] = val_acc
        stats['best_epoch'] = epoch + 1
        if cfg['trach']['save']:
            # Save the checkpoint
            state = {
                'epoch': epoch, 
                'optimizer': optimizer.state_dict(),
                'net': net.state_dict(),
                'stats': stats
            }
            torch.save(state, checkpoint_path)

    # Switch back to training mode
    net.train()

    print('[Epoch: %d]  Train Loss: %.3f   Val Acc: %.3f%%' % ( epoch + 1, training_loss / len(trainloader), val_acc ))
    
    if cfg['track']['save']:
        with open(LOG_FILE, 'a+') as f:
            f.write('%d,%.3f,%.3f\n' % (epoch + 1, training_loss / len(trainloader), val_acc))

    # Step the scheduler after every epoch
    scheduler.step()

end = time.time()
print('Total time trained: {}'.format( str(timedelta(seconds=int(end - start)) ) ))

# Test the model
print('Test Accuracy of the {} on the {} test images: Epoch {}, {} % '.format(network_name, VAL_LEN, stats['best_epoch'], stats['best_acc']))
if cfg['track']['save']:
    with open(LOG_FILE, 'a+') as f:
        f.write('Total time trained: {}\n'.format( str(timedelta(seconds=int(end - start)) ) ))
        f.write('Test Accuracy of the {} on the {} test images: Epoch {}, {} %'.format(network_name, VAL_LEN, stats['best_epoch'], stats['best_acc']))

    with open(RESULT_FILE, 'a+') as f:
        f.write('**********************\n')
        f.write('Results of network {} on dataset {}:\n'.format(network_name, cfg['misc']['dataset']))
        f.write('Accuracy: {}, Epoch: {}, Time: {}\n'.format(stats['best_acc'], stats['best_epoch'], str(timedelta(seconds=int(end - start)) ) ))