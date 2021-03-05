from itertools import repeat
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import argparse
import torch
import hypergrad as hg
import numpy as np
import time
import os
import gc


class CustomTensorIterator:
    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        self.loader = DataLoader(TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs)
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
        return idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', default=50, type=int, help='epoch numbers')
    parser.add_argument('--T', default=10, type=int, help='inner update iterations')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_size', type=int, default=1)
    parser.add_argument('--eta', type=float, default=0.5, help='used in Hessian')
    parser.add_argument('--K', type=int, default=10, help='number of steps to approximate hessian')
    # Only when alg == minibatch, we apply stochastic, otherwise, alg training with full batch
    parser.add_argument('--alg', type=str, default='minibatch', choices=['minibatch', 'reverse', 'fixed_point', 'CG', 'neuman'])
    parser.add_argument('--inner_lr', type=float, default=300.0) # beta
    parser.add_argument('--inner_mu', type=float, default=0.0)
    parser.add_argument('--outer_lr', type=float, default=0.5) # alpha
    parser.add_argument('--outer_mu', type=float, default=0.0)
    parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    args = parser.parse_args()
    # outer_lr, outer_mu = 100.0, 0.0  # nice with 100.0, 0.0 (torch.SGD) tested with T, K = 5, 10 and CG
    # inner_lr, inner_mu = 100., 0.9   # nice with 100., 0.9 (HeavyBall) tested with T, K = 5, 10 and CG
    # parser.add_argument('--seed', type=int, default=0)

    if not args.save_folder:
        args.save_folder = './save_results'
    args.model_name = '{}_bs_{}_vbs_{}_olrmu_{}_{}_ilrmu_{}_{}_eta_{}_T_{}_K_{}'.format(args.alg, 
                       args.batch_size, args.val_size, args.outer_lr, args.outer_mu, args.inner_lr, 
                       args.inner_mu, args.eta, args.T, args.K)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    # parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    # parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    return args


def train_model(args):

    # Constant
    tol = 1e-12
    warm_start = True
    bias = False  # without bias outer_lr can be bigger (much faster convergence)
    train_log_interval = 100
    val_log_interval = 1

    # Basic Setting 
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    cuda = True and torch.cuda.is_available() # torch.cuda.is_available() # cuda=False -> CPU version
    torch.cuda.set_device(1)
    default_tensor_str = 'torch.cuda.FloatTensor' if cuda else 'torch.FloatTensor'
    kwargs = {} # {'num_workers': 1, 'pin_memory': True} if cuda else {}
    torch.set_default_tensor_type(default_tensor_str)
    # torch.multiprocessing.set_start_method('forkserver')

    # Functions 
    def frnp(x): return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float() # from numpy
    def tonp(x, cuda=cuda): return x.detach().cpu().numpy() if cuda else x.detach().numpy() # to numpy
    def train_loss(params, hparams, data):
        x_mb, y_mb = data
        # print(x_mb.size()) = torch.Size([5657, 130107])
        out = out_f(x_mb,  params)
        return F.cross_entropy(out, y_mb) + reg_f(params, *hparams)
    def val_loss(opt_params, hparams):
        x_mb, y_mb = next(val_iterator)
        # print(x_mb.size()) = torch.Size([5657, 130107])
        out = out_f(x_mb,  opt_params[:len(parameters)])
        val_loss = F.cross_entropy(out, y_mb)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(y_mb.view_as(pred)).sum().item() / len(y_mb)

        val_losses.append(tonp(val_loss))
        val_accs.append(acc)
        return val_loss
    def reg_f(params, l2_reg_params, l1_reg_params=None):
        r = 0.5 * ((params[0] ** 2) * torch.exp(l2_reg_params.unsqueeze(1) * ones_dxc)).mean()
        if l1_reg_params is not None:
            r += (params[0].abs() * torch.exp(l1_reg_params.unsqueeze(1) * ones_dxc)).mean()
        return r
    def out_f(x, params):
        out = x @ params[0]
        out += params[1] if len(params) == 2 else 0
        return out
    # evaluate
    def eval(params, x, y):
        out = out_f(x,  params)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)

        return loss, acc

    
    # load twentynews and preprocess
    val_size_ratio = 0.5
    X, y = fetch_20newsgroups_vectorized(subset='train', return_X_y=True,
                                        #remove=('headers', 'footers', 'quotes')
                                        )
    x_test, y_test = fetch_20newsgroups_vectorized(subset='test', return_X_y=True,
                                                #remove=('headers', 'footers', 'quotes')
                                                )
    x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size_ratio)
    print('dataset information:')
    print(x_train.shape, x_val.shape,x_test.shape)
    print(y_train.shape, y_val.shape,y_test.shape)
    # clip feature space here
    x_train = x_train[:,60000:60850]
    x_val = x_val[:,60000:60850]
    x_test = x_test[:,60000:60850]

    train_samples, n_features = x_train.shape
    test_samples, n_features = x_test.shape
    val_samples, n_features = x_val.shape

    n_classes = np.unique(y_train).shape[0]
    # train_samples=5657, val_samples=5657, test_samples=7532, n_features=130107, n_classes=20
    print('Dataset 20newsgroup, train_samples=%i, val_samples=%i, test_samples=%i, n_features=%i, n_classes=%i'
        % (train_samples, val_samples, test_samples, n_features, n_classes))
    ys = [frnp(y_train).long(), frnp(y_val).long(), frnp(y_test).long()]
    xs = [x_train, x_val, x_test]

    if cuda:
        xs = [from_sparse(x).cuda() for x in xs]
    else:
        xs = [from_sparse(x) for x in xs]

    # x_train.size() = torch.Size([5657, 130107])
    # y_train.size() = torch.Size([5657])
    x_train, x_val, x_test = xs
    y_train, y_val, y_test = ys
    
    # torch.DataLoader has problems with sparse tensor on GPU    
    iterators, train_list, val_list = [], [], []


    # For minibatch method, we build the list to store the splited tensor
    if args.alg == 'minibatch':
        for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
            iterators.append(CustomTensorIterator([x, y], batch_size=args.batch_size, shuffle=True, **kwargs))
        train_iterator, val_iterator = iterators
        for _ in range(train_samples // args.batch_size+1):
            train_list.append(next(train_iterator))
        for _ in range(val_samples // args.val_size+1):
            val_list.append(next(val_iterator))
        train_list_len, val_list_len = len(train_list), len(val_list)

        # set up another train_iterator & val_iterator to make sure train_list and val_list are full
        iterators = []
        for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
            iterators.append(repeat([x, y]))
        train_iterator, val_iterator = iterators
    else:
        for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
            iterators.append(repeat([x, y]))
        train_iterator, val_iterator = iterators
       

    # Initialize parameters
    l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
    l1_reg_params = (0.*torch.ones(1)).requires_grad_(True)  # one l1 hp only (best when really low)
    #l2_reg_params = (-20.*torch.ones(1)).requires_grad_(True)  # one l2 hp only (best when really low)
    #l1_reg_params = (-1.*torch.ones(n_features)).requires_grad_(True)
    hparams = [l2_reg_params] # x = hparams[0]
    print('x:',hparams[0].size())
    # hparams: the outer variables (or hyperparameters)
    ones_dxc = torch.ones(n_features, n_classes)

    outer_opt = torch.optim.SGD(lr=args.outer_lr, momentum=args.outer_mu, params=hparams)
    # outer_opt = torch.optim.Adam(lr=0.01, params=hparams)

    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    w = torch.zeros(n_features, n_classes).requires_grad_(True)
    parameters = [w] # y = parameters[0]
    print('y:',parameters[0].size())

    # params_history: the inner iterates (from first to last)
    if bias:
        b = torch.zeros(n_classes).requires_grad_(True)
        parameters.append(b)
 

    total_time = 0
    loss_acc_time_results = np.zeros((args.n_steps+1, 3))
    test_loss, test_acc = eval(parameters, x_test, y_test)
    loss_acc_time_results[0, 0] = test_loss
    loss_acc_time_results[0, 1] = test_acc
    loss_acc_time_results[0, 2] = 0.0
    

    x_prev = hparams.copy()
    y_prev = parameters.copy()
    
    for o_step in range(args.n_steps):
        train_index_list = torch.randperm(train_list_len)
        val_index_list = torch.randperm(val_list_len)
        start_time = time.time()
        #  x -> hparas
        #  y -> parameters
        if args.alg == 'minibatch':
            # hy => hg in STABLE
            loss_train = train_loss(parameters, hparams, train_list[train_index_list[0]])
            hy = torch.autograd.grad(loss_train, parameters, retain_graph=True, create_graph=True)[0]
            hy = torch.reshape(hy, [-1])
            hx = torch.autograd.grad(loss_train, hparams, retain_graph=True, create_graph=True)[0]
            hx = torch.reshape(hx,[-1])

            hyy_k = eval_hessian(hy, parameters)
            hxy_k = eval_hessian(hx, parameters)
            # print('test hessian_xy',hxy_k.shape)
            
            # compute hyy_k-1 and hxy_k-1
            loss_train_k_1 = train_loss(y_prev, x_prev, train_list[train_index_list[0]])
            hy_k_1 = torch.autograd.grad(loss_train_k_1, y_prev, retain_graph=True, create_graph=True)[0]
            hy_k_1 = torch.reshape(hy_k_1, [-1])
            hx_k_1 = torch.autograd.grad(loss_train_k_1, x_prev, retain_graph=True, create_graph=True)[0]
            hx_k_1 = torch.reshape(hx_k_1, [-1])

            hyy_k_1 = eval_hessian(hy_k_1, y_prev)
            hxy_k_1 = eval_hessian(hx_k_1, y_prev)
            
            ''' add projection here '''
            # update Hxy via 12a
            Hxy_k = (1-0.5)*(Hxy_k.detach()-hxy_k_1.detach()) + hxy_k.detach() if o_step > 0  else torch.clone(hxy_k)
            # update Hyy via 12b
            Hyy_k = (1-0.5)*(Hyy_k.detach() - hyy_k_1.detach()) + hyy_k.detach() if o_step > 0  else torch.clone(hyy_k) 

            # compute fx_grad, fy_grad
            fy_gradient = gradient_fy(args, parameters, val_list[val_index_list[1]])
            fy_gradient = torch.reshape(fy_gradient, [-1]).detach()

            # update x and y via 11
            # save x_k-1
            x_prev = [None]
            x_prev[0] = torch.clone(hparams[0]).requires_grad_(True) 
            # 11a
            hparams[0] = hparams[0] - args.outer_lr*(-torch.matmul(-Hxy_k.detach(), torch.matmul(torch.inverse(Hyy_k).detach(), fy_gradient.detach()) ))
            # save y_k-1
            y_prev = [None]
            y_prev[0] = torch.clone(parameters[0]).requires_grad_(True)
            # 11b
            y_vec = torch.reshape(parameters[0],[-1])
            y_vec = y_vec - args.inner_lr*hy - torch.matmul(torch.inverse(Hyy_k).detach(), torch.matmul(Hxy_k.T.detach(),(hparams[0] - x_prev[0]).detach()))
            parameters[0] = torch.reshape(y_vec,(parameters[0].shape[0],parameters[0].shape[1]))
            final_params = parameters
            val_loss(final_params, hparams)
            torch.cuda.empty_cache()
            print('-'*30)

        iter_time = time.time() - start_time
        total_time += iter_time
        if o_step % val_log_interval == 0 or o_step == args.T-1:
            test_loss, test_acc = eval(final_params[:len(parameters)], x_test, y_test)
            loss_acc_time_results[o_step+1, 0] = test_loss
            loss_acc_time_results[o_step+1, 1] = test_acc
            loss_acc_time_results[o_step+1, 2] = total_time
            print('o_step={} ({:.2e}s) Val loss: {:.4e}, Val Acc: {:.2f}%'.format(o_step, iter_time, val_losses[-1],
                                                                                100*val_accs[-1]))
            print('          Test loss: {:.4e}, Test Acc: {:.2f}%'.format(test_loss, 100*test_acc))
            print('          l2_hp norm: {:.4e}'.format(torch.norm(hparams[0])))
            print('-'*30)
            if len(hparams) == 2:
                print('          l1_hp : ', torch.norm(hparams[1]))

    # loop done
    # save result
    file_name = 'results.npy'
    file_addr = os.path.join(args.save_folder, file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_acc_time_results)   

    print(loss_acc_time_results)
    np.savetxt('STABLE.txt',loss_acc_time_results)
    print('HPO ended in {:.2e} seconds\n'.format(total_time))



def from_sparse(x):
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def gradient_fy(args, parameters, data):
    images, labels = data
    out = out_f(images,  parameters)
    #print(out.shape)
    loss = F.cross_entropy(out, labels)
    grad = torch.autograd.grad(loss, parameters)[0]
    return grad


def train_loss(params, hparams, data):
    x_mb, y_mb = data
    out = out_f(x_mb,  params)
    return F.cross_entropy(out, y_mb) + reg_f(params, *hparams)

def gradient_gy(args, parameters, data, weight):
    train_loss_ = train_loss(parameters, weight, data)
    grad = torch.autograd.grad(train_loss_, parameters, create_graph=True)[0]
    return grad

def val_loss(opt_params, hparams):
    x_mb, y_mb = next(val_iterator)
    out = out_f(x_mb,  opt_params[:len(parameters)])
    val_loss = F.cross_entropy(out, y_mb)
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(y_mb.view_as(pred)).sum().item() / len(y_mb)

    val_losses.append(tonp(val_loss))
    val_accs.append(acc)
    return val_loss

def reg_f(params, l2_reg_params, l1_reg_params=None):
    ones_dxc = torch.ones(params[0].size())
    r = 0.5 * ((params[0] ** 2) * torch.exp(l2_reg_params.unsqueeze(1) * ones_dxc)).mean()
    if l1_reg_params is not None:
        r += (params[0].abs() * torch.exp(l1_reg_params.unsqueeze(1) * ones_dxc)).mean()
    return r

def out_f(x, params):
    out = x @ params[0]
    out += params[1] if len(params) == 2 else 0
    return out
# torch.cat -> index copy
def eval_hessian(loss_grad, params):
    # cnt = 0
    print('loss_grad: ',loss_grad.size())
    # g_vector = loss_grad
    '''
    g_count = 0
    for g in loss_grad:
        # loss_grad = g.contiguous().view(-1) if cnt == 0 else torch.cat([loss_grad, g.contiguous().view(-1)])
        # cnt = 1
        loss_grad[g_count] = g.contiguous().view(-1)
        g_count += 1 # g_cound => loss_grad.size()
    # print('loss_grad:',loss_grad.size())
    if torch.all(loss_grad == loss_grad) == True:
        print('very useless')
    '''
    l = loss_grad.size(0)
    # print('xd:', l)
    l2 = torch.reshape(params[0],[-1]).size(0)
    # print('yd:', l2)
    hessian = torch.zeros(l, l2).requires_grad_(False)
    # print('hessian: ',hessian.size())
    for idx in range(l):
        grad2rd = torch.autograd.grad(loss_grad[idx], params,retain_graph=True)
        '''
        cnt = 0
        g2 = torch.zeros(hessian[0].size())
        # print('g2:', g2.size())
        for g in grad2rd:
            # g2_old = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2_old, g.contiguous().view(-1)])
            # cnt = 1
            # print('g contiguous: ',g.contiguous().view(-1).size(),';g2_old:', g2.size(),';g2:', g2.size())
            g2 = g.contiguous().view(-1)
            cnt += 1
            if cnt >1 :
                print('cnt',cnt)
        if torch.all(g2 == torch.reshape(grad2rd[0],[-1])) != True:
            print('sometimes useful')
        '''
        # print('g2: ',g2.size(),'; hessian[index]:', hessian[idx].size(), 'g2_old:',g2_old.size())
        hessian[idx] = torch.reshape(grad2rd[0],[-1]).detach()
    del grad2rd
    return hessian # .cpu().data.numpy() # hessian.cpu().data.numpy()


def main():
    args = parse_args()
    print(args)
    train_model(args)

if __name__ == '__main__':
    main()
