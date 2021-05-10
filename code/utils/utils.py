import os
import json
import logging

import numpy as np

import torch
import torch
import torch.nn.functional as F
import numpy as np
import math

import time
import sys
from scipy.special import lambertw as lambertw_np
# lambertw is not implemented in pytorch 
class LabelDict():
    def __init__(self, dataset='cifar-10'):
        self.dataset = dataset
        if dataset == 'cifar-10':
            self.label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 
                         4: 'deer',     5: 'dog',        6: 'frog', 7: 'horse',
                         8: 'ship',     9: 'truck'}

        self.class_dict = {v: k for k, v in self.label_dict.items()}

    def label2class(self, label):
        assert label in self.label_dict, 'the label %d is not in %s' % (label, self.dataset)
        return self.label_dict[label]

    def class2label(self, _class):
        assert isinstance(_class, str)
        assert _class in self.class_dict, 'the class %s is not in %s' % (_class, self.dataset)
        return self.class_dict[_class]

def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)

    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

def one_hot(ids, n_class):
    # --------------------- 

    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """

    assert len(ids.shape) == 1, 'the ids should be 1-D'
    # ids = torch.LongTensor(ids).view(-1,1) 

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor
    
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model, file_name):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def count_parameters(model):
    # copy from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    # baldassarre.fe's reply
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# Reimplementation of scipy version of lambertw for branching factor = 0
import torch
import math
import warnings

OMEGA = 0.56714329040978387299997  # W(1, 0)
EXPN1 = 0.36787944117144232159553  # exp(-1)

def evalpoly(coeff, degree, z): 
    powers = torch.arange(degree,-1,-1).float().to(z.device)
    return ((z.unsqueeze(-1)**powers)*coeff).sum(-1)

def lambertw(z0, tol=1e-5): 
    # this is a direct port of the scipy version for the 
    # k=0 branch for *positive* z0 (z0 >= 0)

    # skip handling of nans
    if torch.isnan(z0).any(): 
        raise NotImplementedError

    w0 = z0.new(*z0.size())
    # under the assumption that z0 >= 0, then I_branchpt 
    # is never used. 
    I_branchpt = torch.abs(z0 + EXPN1) < 0.3
    I_pade0 = (-1.0 < z0)*(z0 < 1.5)
    I_asy = ~(I_branchpt | I_pade0)
    if I_pade0.any(): 
        z = z0[I_pade0]
        num = torch.Tensor([
            12.85106382978723404255,
            12.34042553191489361902,
            1.0
        ]).to(z.device)
        denom = torch.Tensor([
            32.53191489361702127660,
            14.34042553191489361702,
            1.0
        ]).to(z.device)
        w0[I_pade0] = z*evalpoly(num,2,z)/evalpoly(denom,2,z)

    if I_asy.any(): 
        z = z0[I_asy]
        w = torch.log(z)
        w0[I_asy] = w - torch.log(w)

    # split on positive and negative, 
    # and ignore the divergent series case (z=1)
    w0[z0 == 1] = OMEGA
    I_pos = (w0 >= 0)*(z0 != 1)
    I_neg = (w0 < 0)*(z0 != 1)
    if I_pos.any(): 
        w = w0[I_pos]
        z = z0[I_pos]
        for i in range(100): 
            # positive case
            ew = torch.exp(-w)
            wewz = w - z*ew
            wn = w - wewz/(w + 1 - (w + 2)*wewz/(2*w + 2))

            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_pos] = w

    if I_neg.any(): 
        w = w0[I_neg]
        z = z0[I_neg]
        for i in range(100):
            ew = torch.exp(w)
            wew = w*ew
            wewz = wew - z
            wn = w - wewz/(wew + ew - (w + 2)*wewz/(2*w + 2))
            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_neg] = wn
    return w0




TYPE_2NORM = '2norm'
TYPE_CONJUGATE = 'conjugate'
TYPE_PLAN = 'plan'

MAX_FLOAT = 1e38# 1.7976931348623157e+308

def any_nan(X): 
    return (X != X).any().item()
def any_inf(X): 
    return (X == float('inf')).any().item()

def lamw_np(x): 
    cuda = x.is_cuda
    x = x.cpu().numpy()
    I = x > 1e-10
    y = np.copy(x)
    y[I] = np.real(lambertw_np(x[I]))
    out = torch.from_numpy(np.real(y))
    if cuda: 
        out = out.cuda()
    return out

def lamw(x): 
    I = x > 1e-10
    y = torch.clone(x)
    y[I] = lambertw(x[I])
    return y

# batch dot product
def _bdot(X,Y): 
    return torch.matmul(X.unsqueeze(-2), Y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

def _unfold(x, kernel_size, padding=None): 
    # this is necessary because unfold isn't implemented for multidimensional batches
    size = x.size()
    if len(size) > 4: 
        x = x.contiguous().view(-1, *size[-3:])
    out = F.unfold(x, kernel_size, padding=kernel_size//2)
    if len(size) > 4: 
        out = out.view(*size[:-3], *out.size()[1:])
    return out


def _mm(A,x, shape): 
    kernel_size = A.size(-1)
    nfilters = shape[1]
    unfolded = _unfold(x, kernel_size, padding=kernel_size//2).transpose(-1,-2)
    unfolded = _expand(unfolded, (A.size(-3),A.size(-2)*A.size(-1))).transpose(-2,-3)
    out = torch.matmul(unfolded, collapse2(A.contiguous()).unsqueeze(-1)).squeeze(-1)

    return unflatten2(out)

def wasserstein_cost(X, p=2, kernel_size=5):
    if kernel_size % 2 != 1: 
        raise ValueError("Need odd kernel size")
        
    center = kernel_size // 2
    C = X.new_zeros(kernel_size,kernel_size)
    for i in range(kernel_size): 
        for j in range(kernel_size): 
            C[i,j] = (abs(i-center)**2 + abs(j-center)**2)**(p/2)
    return C

def unsqueeze3(X):
    return X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def collapse2(X): 
    return X.view(*X.size()[:-2], -1)

def collapse3(X): 
    return X.view(*X.size()[:-3], -1)

def _expand(X, shape): 
    return X.view(*X.size()[:-1], *shape)

def unflatten2(X): 
    # print('unflatten2', X.size())
    n = X.size(-1)
    k = int(math.sqrt(n))
    return _expand(X,(k,k))

def _expand_filter(X, nfilters): 
    sizes = list(-1 for _ in range(X.dim()))
    sizes[-3] = nfilters
    return X.expand(*sizes)


def bdot(x,y): 
    return _bdot(collapse3(x), collapse3(y))

def projected_sinkhorn(*args, **kwargs): 
    return log_sinkhorn(*args, objective='2norm', **kwargs)

def conjugate_sinkhorn(*args, **kwargs): 
    return log_sinkhorn(*args, objective='conjugate', **kwargs)

def log_sinkhorn(X, Y, C, epsilon, lam, verbose=False, plan=False,
   objective='2norm', maxiters=50, return_objective=False):  
    """ 
    if objective == '2norm': 
        minimize_Z ||Y-Z||_2 subject to Z in Wasserstein ball around X 

        we return Z

    if objective == 'conjugate': 
        minimize_Z -Y^TZ subject to Z in Wasserstein ball around X

        however instead of Z we return the dual variables u,psi for the 
        equivalent objective: 
        minimize_{u,psi} 

    Inputs: 
        X : batch size x nfilters x image width x image height
        Y : batch size x noutputs x total input dimension

        these really out to be broadcastable

        Same as above but in log space. Note that the implementation follows the 
        rescaled version -\lambda Y^TZ + entropy(W) instead of 
        -Y^TZ + 1/lambda * entropy(W), so the final objective is downscaled by lambda
    """
    batch_sizes = X.size()[:-3]
    nfilters = X.size(-3)

    # size check
    for xd,yd in (zip(reversed(X.size()),reversed(Y.size()))): 
        assert xd == yd or xd == 1 or yd == 1

    # helper functions
    expand3 = lambda x: _expand(x, X.size()[-3:])
    expand_filter = lambda x: _expand_filter(x, X.size(-3))
    mm = lambda A,x: _mm(expand_filter(A),x,X.size())
    norm = lambda x: torch.norm(collapse3(x), dim=-1)
    # like numpy
    allclose = lambda x,y: (x-y).abs() <= 1e-4 + 1e-4*y.abs()
    
    # assert valid distributions
    assert (X>=0).all()
    assert ((collapse3(X).sum(-1) - 1).abs() <= 1e-4).all()
    assert X.dim() == Y.dim()


    size = tuple(max(sx,sy) for sx,sy in zip(X.size(), Y.size()))

    # total dimension size for each example
    m = collapse3(X).size(-1)
    
    if objective == TYPE_CONJUGATE: 
        alpha = torch.log(X.new_ones(*size)/m) + 0.5
        exp_alpha = torch.exp(-alpha)
        beta = -lam*Y.expand_as(alpha).contiguous()
        exp_beta = torch.exp(-beta)

        # check for overflow
        if (exp_beta == float('inf')).any(): 
            print(beta.min())
            raise ValueError('Overflow error: in logP_sinkhorn for e^beta')


        # EARLY TERMINATION CRITERIA: if the nu_1 and the 
        # center of the ball have no pixels with overlapping filters, 
        # thenthe wasserstein ball has no effect on the objective. 
        # Consequently, we should just return the objective 
        # on the center of the ball. Notably, if the filters don't overlap, 
        # then the pixels themselves don't either, so we can conclude that 
        # the objective is 0. 

        # We can detect overlapping filters by applying the cost 
        # filter and seeing if the sum is 0 (e.g. X*C*Y)
        C_tmp = C.clone() + 1
        while C_tmp.dim() < Y.dim(): 
            C_tmp = C_tmp.unsqueeze(0)
        I_nonzero = bdot(X,mm(C_tmp,Y)) != 0
        I_nonzero_ = unsqueeze3(I_nonzero).expand_as(alpha)

        def eval_obj(alpha, exp_alpha, psi, K): 
            return -psi*epsilon - bdot(torch.clamp(alpha,max=MAX_FLOAT),X) - bdot(exp_alpha, mm(K, exp_beta))

        def eval_z(alpha, exp_alpha, psi, K): 
            return exp_beta*mm(K, exp_alpha)

        psi = X.new_ones(*size[:-3])
        K = torch.exp(-unsqueeze3(psi)*C - 1)

        old_obj = -float('inf')
        i = 0

        with torch.no_grad(): 
            while True: 
                alpha[I_nonzero_] = (torch.log(mm(K,exp_beta)) - torch.log(X))[I_nonzero_]
                exp_alpha = torch.exp(-alpha)

                dpsi = -epsilon + bdot(exp_alpha,mm(C*K,exp_beta))
                ddpsi = -bdot(exp_alpha,mm(C*C*K,exp_beta))
                delta = dpsi/ddpsi

                psi0 = psi
                t = X.new_ones(*delta.size())
                neg = (psi - t*delta < 0)
                while neg.any() and t.min().item() > 1e-2:
                    t[neg] /= 2
                    neg = psi - t*delta < 0
                psi[I_nonzero] = torch.clamp(psi - t*delta, min=0)[I_nonzero]

                K = torch.exp(-unsqueeze3(psi)*C - 1)

                # check for convergence
                obj = eval_obj(alpha, exp_alpha, psi, K)
                if verbose: 
                    print('obj', obj)
                i += 1
                if i > maxiters or allclose(old_obj,obj).all(): 
                    if verbose: 
                        print('terminate at iteration {}'.format(i))
                    break

                old_obj = obj

        if return_objective: 
            obj = -bdot(X,Y)
            obj[I_nonzero] = eval_obj(alpha, exp_alpha, psi, K)[I_nonzero]
            return obj
        else: 
            z = eval_z(alpha, exp_alpha, psi,K)
            z[~I_nonzero] = 0
            return z

    elif objective == TYPE_2NORM: 
        alpha = torch.log(X.new_ones(*size)/m)
        beta = torch.log(X.new_ones(*size)/m)
        exp_alpha = torch.exp(-alpha)
        exp_beta = torch.exp(-beta)

        psi = X.new_ones(*size[:-3])
        K = torch.exp(-unsqueeze3(psi)*C - 1)

        def eval_obj(alpha, beta, exp_alpha, exp_beta, psi, K): 
            return (-0.5/lam*bdot(beta,beta) - psi*epsilon 
                    - bdot(torch.clamp(alpha,max=1e10),X) 
                    - bdot(torch.clamp(beta,max=1e10),Y)
                    - bdot(exp_alpha, mm(K, exp_beta)))

        old_obj = -float('inf')
        i = 0

        if verbose:
            start_time = time.time()

        with torch.no_grad(): 
            while True: 
                alphat = norm(alpha)
                betat = norm(beta)

                alpha = (torch.log(mm(K,exp_beta)) - torch.log(X))
                exp_alpha = torch.exp(-alpha)
                
                beta = lamw(lam*torch.exp(lam*Y)*mm(K,exp_alpha)) - lam*Y
                exp_beta = torch.exp(-beta)

                dpsi = -epsilon + bdot(exp_alpha,mm(C*K,exp_beta))
                ddpsi = -bdot(exp_alpha,mm(C*C*K,exp_beta))
                delta = dpsi/ddpsi

                psi0 = psi
                t = X.new_ones(*delta.size())
                neg = (psi - t*delta < 0)
                while neg.any() and t.min().item() > 1e-2:
                    t[neg] /= 2
                    neg = psi - t*delta < 0
                psi = torch.clamp(psi - t*delta, min=0)

                # update K
                K = torch.exp(-unsqueeze3(psi)*C - 1)

                # check for convergence
                obj = eval_obj(alpha, exp_alpha, beta, exp_beta, psi, K) 
                i += 1
                if i > maxiters or allclose(old_obj,obj).all(): 
                    if verbose: 
                        print('terminate at iteration {}'.format(i), maxiters)
                        if i > maxiters: 
                            print('warning: took more than {} iters'.format(maxiters))
                    break
                old_obj = obj
        return beta/lam + Y
        
import os, torch 
import sys
import time

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
