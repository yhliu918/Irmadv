import argparse
import numpy as np
import torch
import os
import sys
import pdb
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torch import nn, optim, autograd
from config import config, args
import foolbox

sys.path.append("../../base_model")
from cifar_resnet18 import cifar_resnet18
from MnistModel import mnist_net

sys.path.append("../")
from attack_method import PGD, EAD, DDN,SaltandPepper,additivenoise,lyh_CW,DF,Rays
from utils import tensor2cuda, numpy2cuda, evaluate, save_model
from collections import OrderedDict

import numpy as np

best_acc = 0  # best test accuracy
best_epoch = -1  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
device = torch.device('cuda:{}'.format(args.d))
if args.dataset == 'cifar10':
    model = cifar_resnet18().to(device)
if args.dataset == 'mnist':
    model = mnist_net().to(device)

# Optimizer
if config.adam:
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
else:
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)

print('==> Preparing data..')
if args.dataset == 'cifar10':
    transform_train = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                             (4, 4, 4, 4), mode='constant', value=0).squeeze()),
        tv.transforms.ToPILImage(),
        tv.transforms.RandomCrop(32),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
    ])
    tr_dataset = tv.datasets.CIFAR10('../datasets/cifar10',
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    te_dataset = tv.datasets.CIFAR10('../datasets/cifar10',
                                     train=False,
                                     transform=tv.transforms.ToTensor(),
                                     download=True)

if args.dataset == 'mnist':
    tr_dataset = tv.datasets.MNIST(root='../datasets/mnist', train=True, transform=tv.transforms.ToTensor(),
                                   download=True)
    te_dataset = tv.datasets.MNIST(root='../datasets/mnist', train=False, transform=tv.transforms.ToTensor(),
                                   download=True)
tr_loader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
te_loader = DataLoader(te_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

# Load checkpoint.
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Define loss function helpers  
def mean_nll(logits, y):
    return F.cross_entropy(logits, y)


def mean_accuracy(logits, y):
    _, preds = logits.max(1)
    correct = preds.eq(y).sum().item()
    return (correct / config.batch_size) * 100.0


def penalty(logits, y):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


# Prepare epsilon
if args.dataset == 'cifar10':
    epsilonl1 = float(2000 / 255)
    epsilonl2 = 0.005
    epsilonlinf = 0.031
if args.dataset == 'mnist':
    epsilonl1 = args.l1
    epsilonl2 = args.l2
    epsilonlinf = args.linf

# Prepare attack
print('==> Preparing attack..')
attacks = [
    
    PGD(model, epsilonlinf, max_iters=40, device=device, _type='linf'),
    PGD(model, epsilonl2, max_iters=100, device=device, _type='l2'),
    PGD(model, epsilonl1, max_iters=100, device=device, _type='l1'),
]

test_attacks = [
    
    #SaltandPepper(model, epsilonl2, max_iters=100, device=device, _type='l2'),
    PGD(model, epsilonlinf, max_iters=40, device=device, _type='linf'),
    PGD(model, epsilonl2, max_iters=100, device=device, _type='l2'),
    PGD(model, epsilonl1, max_iters=100, device=device, _type='l1'),
    #Rays(model, epsilonlinf, max_iters=40, device=device, _type='linf'),
    #DF(model, epsilonlinf, max_iters=40, device=device, _type='linf'),
    #EAD(model, epsilonl1, max_iters=100, device=device, _type='l1'),
    lyh_CW(model, epsilonl2, max_iters=100, device=device, _type='l2'),
    DDN(model, epsilonl2, max_iters=100, device=device, _type='l2'),
    additivenoise(model, epsilonl2, max_iters=100, device=device, _type='l1'),
    additivenoise(model, epsilonl2, max_iters=100, device=device, _type='l2'),
]
penal_list = dict()
for j in range(len(attacks)):
    penal_list.setdefault(j, [])
for j in range(len(attacks)):
    for i in range(config.total_epochs):
        penal_list[j].append(config.attack_penalty[j])


# Train loop

def train(epoch):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    train_penalty = 0
    pbar = tqdm(tr_loader)

    for g in optimizer.param_groups:
        current_lr = g['lr']
        break
    pbar.set_description('Train epoch: {}/{} lr: {:.1e} '.format(epoch, config.total_epochs, current_lr))

    for batch_idx, (data, label) in enumerate(pbar):

        data, label = data.to(device), label.to(device)
        all_loss = torch.zeros(len(attacks)).to(device)
        
        optimizer.zero_grad()
        total_loss = 0
        total_penalty = 0
        penalty_loss_list = []
        losses=[]
        # clean accuracy
        output = model(data)
        loss = mean_nll(output, label)
        total_loss += loss
        losses.append(loss)
        # attack accuracy
        for att_idx, attack in enumerate(attacks):
            adv_data = attack.perturb(data, label, True)[0]
            output = model(adv_data)
            loss = mean_nll(output, label)

            # accuracy = mean_accuracy(output, label)
            tmp_penalty = penalty(output, label)
            penalty_loss_list.append(tmp_penalty)
            losses.append(loss)

            total_loss += loss
            total_penalty += penal_list[att_idx][epoch] * tmp_penalty
            
        '''
          weight_norm = torch.tensor(0.).to(device)
          for w in model.parameters():
              weight_norm += w.norm().pow(2)
          total_loss += args.l2_regularizer_weight * weight_norm
        '''
        
      
        if args.mode=='MAX':
            total_loss = torch.stack(losses).max()
        if args.mode == 'IRM':
            total_loss += total_penalty
            total_loss/=sum(config.attack_penalty)
        if args.mode == 'VREX':
            variance=torch.stack(losses).var()
            total_loss+=variance*config.var_weight
        if args.mode == 'MMREX':
            total_loss=(1+config.mm_weight*len(attacks))*max(losses)- config.mm_weight*sum(losses)
        
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_penalty += total_penalty.item()

        pbar_dic = OrderedDict()

        # the loss is the mean train_loss up till now

        pbar_dic['loss'] = '{:.3f}'.format(train_loss / (batch_idx + 1))
        # pbar_dic['penalty'] = '{:.3f}'.format(train_penalty / (batch_idx + 1))
        
        pbar_dic['pgd linf'] = '{:.5f}'.format(penalty_loss_list[0])
        pbar_dic['pgd l2'] = '{:.5f}'.format(penalty_loss_list[1])

        pbar.set_postfix(pbar_dic)



def test(epoch):
    global best_acc
    global best_epoch
    model.eval()
    _loss = dict()
    _accuracy = dict()
    # _penalty=dict()
    pbar = tqdm(te_loader)
    max_acc=0.
    test_num=0
    for batch_idx, (inputs, targets) in enumerate(pbar):
        data, label = inputs.to(device), targets.to(device)
        with torch.no_grad():
            output = model(data)
            loss = mean_nll(output, label)
            accuracy = mean_accuracy(output, label)
            _loss.setdefault('clean', []).append(loss.cpu().item())
            _accuracy.setdefault('clean', []).append(accuracy)
        adv_list=[]
        for attack in test_attacks:
            adv_data = attack.perturb(data, label, True)[0].detach()
            if attack.name=='RayS':
                succ=attack.perturb(data, label, True)[1].detach()
                adv_list.append(succ.cpu().float())
                accuracy=(succ.cpu().float()).sum()
                _accuracy.setdefault(str(attack.name) + str(attack._type), []).append(accuracy)
                _loss.setdefault(str(attack.name) + str(attack._type), []).append(0)
                continue
            with torch.no_grad():
                output = model(adv_data)
                _, preds = output.max(1)
                loss = mean_nll(output, label)
                adv_list.append(preds.eq(label).cpu().float())
                accuracy = mean_accuracy(output, label)

                _loss.setdefault(str(attack.name) + str(attack._type), []).append(loss.cpu().item())
                _accuracy.setdefault(str(attack.name) + str(attack._type), []).append(accuracy)
        
        all_adv=torch.mean(torch.stack(adv_list),0)
        all_adv=(all_adv>=0.99).sum()
        max_acc+=all_adv
        test_num+=1

        if batch_idx>=10:
            max_acc=max_acc/test_num
            break
    acc = 0

    print("*" * 10 + 'loss' + "        " + 'accuracy')
    for keys in _accuracy.keys():
        acc += np.mean(_accuracy[str(keys)])
        print(str(keys) + ':' + str('{:.3f}'.format(np.mean(_loss[str(keys)]))) + "        " + str(
            '{:.3f}'.format(np.mean(_accuracy[str(keys)]))))
            
    acc /= len(_accuracy.keys())
    print('max :' + str('{:.3f}'.format(max_acc)  ))
    print('avg :'+str('{:.3f}'.format(acc) ))
  
    if max_acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': max_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        saveplace = './checkpoint/ckpt_l1'+str(args.l1)+'_l2_'+str(args.l2)+'_linf_'+str(args.linf)+'.pth'
        torch.save(state, saveplace)
        best_acc = max_acc
        best_epoch = epoch


if __name__ == "__main__":

    for epoch in range(start_epoch, config.total_epochs):
        train(epoch)
        if ((epoch + 1) % config.val_interval == 0) :
            test(epoch)

        lr_scheduler.step()

    print("Best Acc {:.2f} at {:} epoch".format(best_acc, best_epoch))
