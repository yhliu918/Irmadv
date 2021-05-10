"""
this code is modified from https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

original author: Utku Ozbulak - github.com/utkuozbulak
"""
import sys

sys.path.append("..")

import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from foolbox import attacks, criteria, models
import advertorch
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack

import pdb

from utils import tensor2cuda
from foolbox.attacks.gradient_descent_base import clip_lp_norms
from foolbox.devutils import atleast_kd

######################### PGD ######################################
class PGD():
    def __init__(self, model, epsilon, max_iters, device, _type='linf'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'PGD'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()

        fmodel = models.PyTorchModel(self.model, device=self.device, bounds=(0, 1))
        epsilons = [self.epsilon]
        if self._type == 'linf':
            attack = attacks.LinfProjectedGradientDescentAttack(steps=self.steps, random_start=random_start)
        if self._type == 'l2':
            attack = attacks.L2ProjectedGradientDescentAttack(steps=self.steps, random_start=random_start)
        if self._type == 'l1':
            attack = attacks.SparseL1DescentAttack(steps=self.steps, random_start=random_start)
        advs, _, success = attack(fmodel, original_images, labels, epsilons=epsilons)
        # criterion = criteria.Misclassification(labels)

        self.model.train()
        return advs

######################### FGSM ######################################
class FastGradient():
    def __init__(self, model, epsilon, max_iters, device, _type='linf'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'FastGradient'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()

        fmodel = models.PyTorchModel(self.model, device=self.device, bounds=(0, 1))

        epsilons = [self.epsilon]
        if self._type == 'linf':
            attack = attacks.LinfFastGradientAttack(random_start=random_start)
        if self._type == 'l2':
            attack = attacks.L2FastGradientAttack(random_start=random_start)

        advs, _, success = attack(fmodel, original_images, labels, epsilons=epsilons)
        # criterion = criteria.Misclassification(labels)

        self.model.train()
        return advs
######################### deepfool ######################################
class DF():
    def __init__(self, model, epsilon, max_iters, device, _type='linf'):
        self.epsilon = epsilon
        self.max_iters = max_iters
        self._type = _type
        self.model = model
        self.name = 'DeepFool'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        attacker = DeepFool(nb_candidate=5, max_iter=40)
        adv_x = attacker.attack(self.model, original_images)
        self.model.train()
        return [adv_x, ]
        
class DeepFool(object):
    def __init__(self, nb_candidate=10, overshoot=0.02, max_iter=50, clip_min=0.0, clip_max=1.0):
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, model, x):
        device = x.device

        with torch.no_grad():
            logits = model(x)
        self.nb_classes = logits.size(-1)
        assert self.nb_candidate <= self.nb_classes, 'nb_candidate should not be greater than nb_classes'

        # preds = logits.topk(self.nb_candidate)[0]
        # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
        # grads will be the shape [batch_size, nb_candidate, image_size]

        adv_x = x.clone().requires_grad_()

        iteration = 0
        logits = model(adv_x)
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        w = torch.squeeze(torch.zeros(x.size()[1:])).to(device)
        r_tot = torch.zeros(x.size()).to(device)
        original = current

        while ((current == original).any and iteration < self.max_iter):
            predictions_val = logits.topk(self.nb_candidate)[0]
            gradients = torch.stack(jacobian(predictions_val, adv_x, self.nb_candidate), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] != original[idx]:
                        continue
                    for k in range(1, self.nb_candidate):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, self.clip_min, self.clip_max).requires_grad_()
            logits = model(adv_x)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1

        adv_x = torch.clamp((1 + self.overshoot) * r_tot + x, self.clip_min, self.clip_max)
        return adv_x


def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives

######################### EAD ######################################
class EAD():
    def __init__(self, model, epsilon, max_iters, device, _type='l1'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'EAD'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        if self._type == 'l1':
            attack = advertorch.attacks.ElasticNetL1Attack(self.model,num_classes=10,max_iterations=1000)
            # adversary = ElasticNetL1Attack(self.model,  num_classes=10,max_iterations=100,decision_rule='L1')
        # advs = adversary.perturb(original_images, labels)
        # criterion = criteria.Misclassification(labels)
        advs = attack.perturb(original_images, labels)
        self.model.train()
        return [advs,]

######################### DDN ###################################### 
class DDN():
    def __init__(self, model, epsilon, max_iters, device, _type='l1'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'DDN'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        fmodel = models.PyTorchModel(self.model, device=self.device, bounds=(0, 1))
        epsilons = [self.epsilon]
        if self._type == 'l2':
            attack = attacks.DDNAttack(init_epsilon= 1.0, steps = 10, gamma = 0.05)
            # adversary = ElasticNetL1Attack(self.model,  num_classes=10,max_iterations=100,decision_rule='L1')
        # advs = adversary.perturb(original_images, labels)
        # criterion = criteria.Misclassification(labels)
        advs, _, success = attack(fmodel, original_images, labels, epsilons=epsilons)
        self.model.train()
        return advs


######################### SaltandPepper ######################################       
class SaltandPepper():
    def __init__(self, model, epsilon, max_iters, device, _type='l1'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'Salt_and_Pepper'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        fmodel = models.PyTorchModel(self.model, device=self.device, bounds=(0, 1))
        epsilons = [self.epsilon]
        if self._type == 'l2':
            attack = attacks.SaltAndPepperNoiseAttack(steps=5,across_channels=False)
            # adversary = ElasticNetL1Attack(self.model,  num_classes=10,max_iterations=100,decision_rule='L1')
        # advs = adversary.perturb(original_images, labels)
        # criterion = criteria.Misclassification(labels)
        advs, _, success = attack(fmodel, original_images, labels, epsilons=epsilons)
        self.model.train()
        return advs

######################### additive noise ######################################
def proj_l1ball(x, epsilon=10, device="cuda:1"):
    assert epsilon > 0
    # compute the vector of absolute values
    u = x.abs()
    xshape = x.shape
    if (u.sum(dim=(1, 2, 3)) <= epsilon).all():
        # check if x is already a solution
        return x

    # y = x* epsilon/norms_l1(x)
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device=device)
    # compute the solution to the original problem on v
    # pdb.set_trace()

    y = y.view(-1, xshape[1], xshape[2], xshape[3])
    y *= x.sign()
    return y

def proj_simplex(v, s=1, device="cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size, 1, -1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending=True)
    cssv = u.cumsum(dim=2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n + 1).float().to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim=2)
    w = (comp - 1).cumsum(dim=2)
    u = u + w
    rho = torch.argmax(u, dim=2)
    rho = rho.view(batch_size)
    c = torch.Tensor([cssv[i, 0, rho[i]] for i in range(cssv.shape[0])]).to(device)
    c = c - s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c, (rho.float() + 1))
    theta = theta.view(batch_size, 1, 1, 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]

def norms_l2(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

class additivenoise():
    def __init__(self, model, epsilon, max_iters, device, var=1., _type='l2'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'additivenoise'
        self.device = device
        self.var = var
        self.mode = 'repeat'  # std/repeat

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        epsilons = [self.epsilon]
        if self._type == 'l2':
            delta = torch.zeros_like(original_images).to(self.device)
            if self.mode == 'repeat':
                for steps in range(self.steps):
                    noise = self.var * torch.randn(original_images.shape).to(self.device)                   
                    delta = delta + noise
                    delta *= self.epsilon / norms_l2(delta)
                    delta = torch.clamp(original_images + delta, 0., 1.) - original_images
                advs = original_images + delta

        if self._type == 'l1':
            delta = torch.zeros_like(original_images).to(self.device)
            if self.mode == 'repeat':
                for steps in range(self.steps):
                    noise = self.var * torch.randn(original_images.shape).to(self.device)
                    delta = delta + noise
                    if (norms_l1(delta) > self.epsilon).any():
                        delta.data = proj_l1ball(delta, self.epsilon, self.device)
                    delta *= proj_l1ball(noise, self.epsilon, self.device)
                    delta = torch.clamp(original_images + delta, 0., 1.) - original_images

                advs = original_images + delta

        self.model.train()
        return [advs,]


######################### Boundary ######################################
class Boundary():
    def __init__(self, model, epsilon, device, max_iters=5000, _type='l2'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'Boundaryattack'
        self.device = device

    def perturb(self, original_images, labels, random_start=True, starting_points=None):
        self.model.eval()
        fmodel = models.PyTorchModel(self.model, device=self.device, bounds=(0, 1))
        epsilons = [self.epsilon]
        if self._type == 'l2':
            attack = attacks.BoundaryAttack(
                init_attack=LinearSearchBlendedUniformNoiseAttack(steps=5000),
                # init_attack=None,
                steps=self.steps, spherical_step=1e-2,
                source_step=1e-2, source_step_convergance=1e-07, step_adaptation=1.5,
                update_stats_every_k=10)
        if starting_points is not None:
            advs, _, success = attack(fmodel, original_images, labels,
                                      epsilons=epsilons,
                                      starting_points=starting_points
                                      )
        else:
            # advs, _, success = attack(fmodel, original_images, labels,
            #                           epsilons=epsilons,
            #                           )
            advs = attack.run(fmodel, original_images,
                                          criteria.Misclassification(labels),
                                          # epsilons=epsilons,
                                          )
            # pdb.set_trace()
        self.model.train()
        return [advs,]

######################### BBA ######################################
class BrendelBethgeAttack():
    def __init__(self, model, epsilon, max_iters, device, _type='linf'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'BrendelBethgeAttack'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        fmodel = models.PyTorchModel(self.model, device=self.device, bounds=(0, 1))
        epsilons = [self.epsilon]
        if self._type == 'linf':
            attack = attacks.LinfinityBrendelBethgeAttack(steps=1000, lr=0.001, lr_decay=0.5, lr_num_decay=20,
                                                          momentum=0.8, binary_search_steps=10)
        if self._type == 'l2':
            attack = attacks.L2BrendelBethgeAttack(steps=0, lr=0.001, lr_decay=0.5, lr_num_decay=20, momentum=0.8,
                                                   binary_search_steps=0)
        if self._type == 'l1':
            attack = attacks.L1BrendelBethgeAttack(steps=100, lr=0.001, lr_decay=0.5, lr_num_decay=20, momentum=0.8,
                                                   binary_search_steps=10)
        if self._type == 'l0':
            attack = attacks.L0BrendelBethgeAttack(steps=1000, lr=0.001, lr_decay=0.5, lr_num_decay=20, momentum=0.8,
                                                   binary_search_steps=10)

        advs, _, success = attack(fmodel, original_images, labels, epsilons=epsilons)

        # criterion = criteria.Misclassification(labels)
        self.model.train()
        return advs


'''
    class CW():
        def __init__(self, model, epsilon, max_iters, device, c=1.0, _type='l2'):
            self.epsilon = epsilon
            self.max_iters = max_iters
            self._type = _type
            self.model = model
            self.name = 'C&W'
            self.device = device
            self.c = c
    
        def perturb(self, original_images, labels, random_start=True):
            self.model.eval()
            advs = cw_l2_attack(self.device, self.model, original_images, labels, c=self.c, kappa=0,
                                max_iter=self.max_iters, learning_rate=0.01)
            self.model.train()
            return [advs, ]
    
    
    def cw_l2_attack(device, model, images, labels, targeted=False, c=1.0, kappa=0, max_iter=1000, learning_rate=0.01):
        images = images.to(device)
        labels = labels.to(device)
    
        # Define f-function
        def f(x):
            outputs = model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
    
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
    
            # If targeted, optimize for making the other class most likely
            if targeted:
                return torch.clamp(i - j, min=-kappa)
    
            # If untargeted, optimize for making the other class most likely
            else:
                return torch.clamp(j - i, min=-kappa)
    
        w = torch.zeros_like(images, requires_grad=True).to(device)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
    
        prev = 1e10
        for step in range(max_iter):
    
            a = 1 / 2 * (nn.Tanh()(w) + 1)
    
            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(c * f(a))
            cost = loss1 + loss2
    
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
    
            # Early Stop when loss does not converge.
            if step % (max_iter // 10) == 0:
                if cost > prev:
                    # print('Attack Stopped due to CONVERGENCE....')
                    return a
                prev = cost
            # print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')
    
        attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
        return attack_images
'''

######################### CW ######################################
from typing import Tuple, Optional
import torch.optim as optim
class lyh_CW():
    def __init__(self, model, epsilon, max_iters, device, c=1.0, _type='l2'):
        self.epsilon = epsilon
        self.max_iters = max_iters
        self._type = _type
        self.model = model
        self.name = 'C&W'
        self.device = device
        self.c = c

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        attacker = CarliniWagnerL2((0.0, 1.0), 10, learning_rate=0.01, search_steps=9, max_iterations=100,
                           initial_const=10, quantize=False, device=self.device)
        adv_x = attacker.attack(self.model, original_images, labels, targeted=False)
        self.model.train()
        return [adv_x, ]
        
class CarliniWagnerL2:
    """
    Carlini's attack (C&W): https://arxiv.org/abs/1608.04644
    Based on https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py

    Parameters
    ----------
    image_constraints : tuple
        Bounds of the images.
    num_classes : int
        Number of classes of the model to attack.
    confidence : float, optional
        Confidence of the attack for Carlini's loss, in term of distance between logits.
    learning_rate : float
        Learning rate for the optimization.
    search_steps : int
        Number of search steps to find the best scale constant for Carlini's loss.
    max_iterations : int
        Maximum number of iterations during a single search step.
    initial_const : float
        Initial constant of the attack.
    quantize : bool, optional
        If True, the returned adversarials will have possible values (1/255, 2/255, etc.).
    device : torch.device, optional
        Device to use for the attack.
    callback : object, optional
        Callback to display losses.
    """

    def __init__(self,
                 image_constraints: Tuple[float, float],
                 num_classes: int=10,
                 confidence: float = 0,
                 learning_rate: float = 0.01,
                 search_steps: int = 9,
                 max_iterations: int = 1000,
                 abort_early: bool = True,
                 initial_const: float = 0.001,
                 quantize: bool = False,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:

        self.confidence = confidence
        self.learning_rate = learning_rate

        self.binary_search_steps = search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.num_classes = num_classes

        self.repeat = self.binary_search_steps >= 10

        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]
        self.boxmul = (self.boxmax - self.boxmin) / 2
        self.boxplus = (self.boxmin + self.boxmax) / 2
        self.quantize = quantize

        self.device = device
        self.callback = callback
        self.log_interval = 10

    @staticmethod
    def _arctanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5

    def _step(self, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor, tinputs: torch.Tensor,
              modifier: torch.Tensor, labels: torch.Tensor, labels_infhot: torch.Tensor, targeted: bool,
              const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = inputs.shape[0]
        adv_input = torch.tanh(tinputs + modifier) * self.boxmul + self.boxplus
        l2 = (adv_input - inputs).view(batch_size, -1).pow(2).sum(1)

        logits = model(adv_input)

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        if targeted:
            # if targeted, optimize for making the other class most likely
            logit_dists = torch.clamp(other - real + self.confidence, min=0)
        else:
            # if non-targeted, optimize for making this class least likely.
            logit_dists = torch.clamp(real - other + self.confidence, min=0)

        logit_loss = (const * logit_dists).sum()
        l2_loss = l2.sum()
        loss = logit_loss + l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return adv_input.detach(), logits.detach(), l2.detach(), logit_dists.detach(), loss.detach()

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model

        """
        batch_size = inputs.shape[0]
        tinputs = self._arctanh((inputs - self.boxplus) / self.boxmul)

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(batch_size, device=self.device)
        CONST = torch.full((batch_size,), self.initial_const, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)

        o_best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        o_best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        o_best_attack = inputs.clone()

        # setup the target variable, we need it to be in one-hot form for the loss function
        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        for outer_step in range(self.binary_search_steps):

            # setup the modifier variable, this is the variable we are optimizing over
            modifier = torch.zeros_like(inputs, requires_grad=True)

            # setup the optimizer
            optimizer = optim.Adam([modifier], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
            best_l2 = torch.full((batch_size,), 1e10, device=self.device)
            best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == (self.binary_search_steps - 1):
                CONST = upper_bound

            prev = float('inf')
            for iteration in range(self.max_iterations):
                # perform the attack
                adv, logits, l2, logit_dists, loss = self._step(model, optimizer, inputs, tinputs, modifier,
                                                                labels, labels_infhot, targeted, CONST)

                if self.callback and (iteration + 1) % self.log_interval == 0:
                    self.callback.scalar('logit_dist_{}'.format(outer_step), iteration + 1, logit_dists.mean().item())
                    self.callback.scalar('l2_norm_{}'.format(outer_step), iteration + 1, l2.sqrt().mean().item())

                # check if we should abort search if we're getting nowhere.
                if self.abort_early and iteration % (self.max_iterations // 10) == 0:
                    if loss > prev * 0.9999:
                        break
                    prev = loss

                # adjust the best result found so far
                predicted_classes = (logits - labels_onehot * self.confidence).argmax(1) if targeted else \
                    (logits + labels_onehot * self.confidence).argmax(1)

                is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
                is_smaller = l2 < best_l2
                o_is_smaller = l2 < o_best_l2
                is_both = is_adv * is_smaller
                o_is_both = is_adv * o_is_smaller

                best_l2[is_both] = l2[is_both]
                best_score[is_both] = predicted_classes[is_both]
                o_best_l2[o_is_both] = l2[o_is_both]
                o_best_score[o_is_both] = predicted_classes[o_is_both]
                o_best_attack[o_is_both] = adv[o_is_both]

            # adjust the constant as needed
            adv_found = (best_score == labels) if targeted else ((best_score != labels) * (best_score != -1))
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10

        if self.quantize:
            adv_found = o_best_score != -1
            o_best_attack[adv_found] = self._quantize(model, inputs[adv_found], o_best_attack[adv_found],
                                                      labels[adv_found], targeted=targeted)

        # return the best solution found
        return o_best_attack

    def _quantize(self, model: nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor,
                  targeted: bool = False) -> torch.Tensor:
        """
        Quantize the continuous adversarial inputs.

        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack.
        adv : torch.Tensor
            Batch of continuous adversarial perturbations produced by the attack.
        labels : torch.Tensor
            Labels of the samples if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be quantized and adversarial to the model.

        """
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.round((adv - inputs) * 255) / 255
        delta.requires_grad_(True)
        logits = model(inputs + delta)
        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        i = 0
        while not is_adv.all() and i < 100:
            loss = F.cross_entropy(logits, labels, reduction='sum')
            grad = autograd.grad(loss, delta)[0].view(batch_size, -1)
            order = grad.abs().max(1, keepdim=True)[0]
            direction = (grad / order).int().float()
            direction.mul_(1 - is_adv.float().unsqueeze(1))
            delta.data.view(batch_size, -1).sub_(multiplier * direction / 255)

            logits = model(inputs + delta)
            is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
            i += 1

        delta.detach_()
        if not is_adv.all():
            delta.data[~is_adv].copy_(torch.round((adv[~is_adv] - inputs[~is_adv]) * 255) / 255)

        return inputs + delta
        
######################### wasserstein ######################################
from utils import conjugate_sinkhorn, projected_sinkhorn
from utils import wasserstein_cost
class wasserstein():
    def __init__(self, model, epsilon, max_iters, device):
        self._type='wasserstein'
        self.max_iters = max_iters
        self.model = model
        self.epsilon=epsilon
        self.name = 'wasserstein'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        adv_x, _, _ = wasserstein_attack(original_images,labels, self.model, epsilon=self.epsilon, epsilon_iters=10, epsilon_factor=1.1, 
           p=2, kernel_size=5, maxiters=100, 
           alpha=0.1, xmin=0, xmax=1, normalize=lambda x: x, verbose=0, 
           regularization=1000, sinkhorn_maxiters=100, 
           ball='wasserstein', norm='l2')
        
        self.model.train()
        return [adv_x, ]
        

def wasserstein_attack(X,y, net, epsilon=0.01, epsilon_iters=10, epsilon_factor=1.1, 
           p=2, kernel_size=5, maxiters=40, 
           alpha=0.1, xmin=0, xmax=1, normalize=lambda x: x, verbose=0, 
           regularization=1000, sinkhorn_maxiters=40, 
           ball='wasserstein', norm='l2'): 
    batch_size = X.size(0)
    epsilon = X.new_ones(batch_size)*epsilon
    C = wasserstein_cost(X, p=p, kernel_size=kernel_size)
    normalization = X.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
    X_ = X.clone()

    X_best = X.clone()
    err_best = err = net(normalize(X)).max(1)[1] != y
    epsilon_best = epsilon.clone()

    t = 0
    while True: 
        X_.requires_grad = True
        opt = optim.SGD([X_], lr=0.1)
        loss = nn.CrossEntropyLoss()(net(normalize(X_)),y)
        opt.zero_grad()
        loss.backward()

        with torch.no_grad(): 
            # take a step
            if norm == 'linfinity': 
                X_[~err] += alpha*torch.sign(X_.grad[~err])
            elif norm == 'l2': 
                X_[~err] += (alpha*X_.grad/(X_.grad.view(X.size(0),-1).norm(dim=1).view(X.size(0),1,1,1)))[~err]
            elif norm == 'wasserstein': 
                sd_normalization = X_.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
                X_[~err] = (conjugate_sinkhorn(X_.clone()/sd_normalization, 
                                               X_.grad, C, alpha, regularization, 
                                               verbose=verbose, maxiters=sinkhorn_maxiters
                                               )*sd_normalization)[~err]
            else: 
                raise ValueError("Unknown norm")

            # project onto ball
            if ball == 'wasserstein': 
                X_[~err] = (projected_sinkhorn(X.clone()/normalization, 
                                          X_.detach()/normalization, 
                                          C,
                                          epsilon,
                                          regularization, 
                                          verbose=verbose, 
                                          maxiters=sinkhorn_maxiters)*normalization)[~err]
            elif ball == 'linfinity': 
                X_ = torch.min(X_, X + epsilon.view(X.size(0), 1, 1,1))
                X_ = torch.max(X_, X - epsilon.view(X.size(0), 1, 1,1))
            else:
                raise ValueError("Unknown ball")
            X_ = torch.clamp(X_, min=xmin, max=xmax)
            
            err = (net(normalize(X_)).max(1)[1] != y)
            err_rate = err.sum().item()/batch_size
            if err_rate > err_best.sum().item()/batch_size:
                X_best = X_.clone() 
                err_best = err
                epsilon_best = epsilon.clone()

            if verbose and t % verbose == 0:
                print(t, loss.item(), epsilon.mean().item(), err_rate)
            
            t += 1
            if err_rate == 1 or t == maxiters: 
                break

            if t > 0 and t % epsilon_iters == 0: 
                epsilon[~err] *= epsilon_factor

    epsilon_best[~err] = float('inf')
    return X_best, err_best, epsilon_best
    
    
######################### Rays ######################################    
from general_torch_model import GeneralTorchModel
from utils import progress_bar
class Rays():
    def __init__(self, model, epsilon, max_iters, device, _type='linf'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'RayS'
        self.device = device

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        torch_model = GeneralTorchModel(self.model, n_class=10, im_mean=None, im_std=None)
        attack = RayS(torch_model, self.device,epsilon=self.epsilon/2)
        adv_x, queries, adbd, succ = attack(original_images, labels,query_limit=10000 )
        
        self.model.train()
        return [adv_x,succ]
        

class RayS(object):
    def __init__(self, model, device,epsilon=0.031, order=np.inf):
        self.model = model
        self.ord = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None
        self.device=device

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).to(self.device)
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out

    def attack_hard_label(self, x, y, target=None, query_limit=10000, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        # init variables
        self.queries = torch.zeros_like(y).to(self.device)
        self.sgn_t = torch.sign(torch.ones(shape)).to(self.device)
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).to(self.device)
        working_ind = (self.d_t > self.epsilon).nonzero().flatten()

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
 
        block_level = 0
        block_ind = 0
        for i in range(query_limit):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < query_limit) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), self.ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > self.epsilon).nonzero().flatten()

            if torch.sum(self.queries >= query_limit) == shape[0]:
                print('out of queries')
                break

            progress_bar(torch.min(self.queries.float()), query_limit,
                         'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d'
                         % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
                            len(working_ind) / len(x), i + 1))
 

        stop_queries = torch.clamp(stop_queries, 0, query_limit)
        return self.x_final, stop_queries, dist, (dist >= self.epsilon)

    # check whether solution is found
    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        if target:
            #return self.model.predict_label(x[mask]) != y[mask]
            return self.model.predict_label(x[mask]) == target[mask]
        else:
            return self.model.predict_label(x[mask]) != y[mask]

    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, target, sgn, valid_mask, tol=1e-3):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().to(self.device)
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]

    def __call__(self, data, label, target=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, query_limit=query_limit)
