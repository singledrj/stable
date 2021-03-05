from itertools import repeat
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_20newsgroups_vectorized
import torchvision
from torchvision import datasets, transforms

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
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--eta', type=float, default=0.5, help='used in Hessian')
    parser.add_argument('--K', type=int, default=10, help='number of steps to approximate hessian')
    # Only when alg == minibatch, we apply stochastic, otherwise, alg training with full batch
    parser.add_argument('--alg', type=str, default='minibatch', choices=['minibatch', 'reverse', 'fixed_point', 'CG', 'neuman'])
    parser.add_argument('--inner_lr', type=float, default=0.1)
    parser.add_argument('--inner_mu', type=float, default=0.0)
    parser.add_argument('--outer_lr', type=float, default=1e-5)
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

    cuda = True and torch.cuda.is_available()
    default_tensor_str = 'torch.cuda.FloatTensor' if cuda else 'torch.FloatTensor'
    kwargs = {} # {'num_workers': 1, 'pin_memory': True} if cuda else {}
    torch.set_default_tensor_type(default_tensor_str)
    # torch.multiprocessing.set_start_method('forkserver')

    # Functions 
    def frnp(x): return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float()
    def tonp(x, cuda=cuda): return x.detach().cpu().numpy() if cuda else x.detach().numpy()
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
    def eval(params, x, y):
        out = out_f(x,  params)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)

        return loss, acc

    '''
    # load twentynews and preprocess
    val_size_ratio = 0.5
    X, y = fetch_20newsgroups_vectorized(subset='train', return_X_y=True,
                                        #remove=('headers', 'footers', 'quotes')
                                        )
    x_test, y_test = fetch_20newsgroups_vectorized(subset='test', return_X_y=True,
                                                #remove=('headers', 'footers', 'quotes')
                                                )
    x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size_ratio)
    
    X,y -> train data & label
    x_test,y_test -> test data & label
    from X,y ====> x_train, x_val, y_train, y_val
    '''

    # load MNIST
    val_size_ratio = 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)
    test_dataset = datasets.MNIST(root="./data/",
                            transform = transform,
                            train = False)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    x,y = next(iter(train_loader))[0],next(iter(train_loader))[1].cpu()
    x = x.view(x.shape[0],-1).cpu()
    # y = y.view(y.shape[0],-1)
    x_test,y_test = next(iter(test_loader))[0],next(iter(test_loader))[1].cpu()
    x_test = x_test.view(x_test.shape[0],-1).cpu()
    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=val_size_ratio)
    x_test = x_test.numpy()
    x_train = x_train.numpy()
    x_val = x_val.numpy()
    y_test = y_test.numpy()
    y_train = y_train.numpy()
    y_val = y_val.numpy()

    train_samples, n_features = x_train.shape
    test_samples, n_features = x_test.shape
    val_samples, n_features = x_val.shape
    n_classes = np.unique(y_train).shape[0]
    # train_samples=5657, val_samples=5657, test_samples=7532, n_features=130107, n_classes=20
    print('Dataset 20newsgroup, train_samples=%i, val_samples=%i, test_samples=%i, n_features=%i, n_classes=%i'
        % (train_samples, val_samples, test_samples, n_features, n_classes))
    ys = [frnp(y_train).long(), frnp(y_val).long(), frnp(y_test).long()]
    xs = [frnp(x_train), frnp(x_val), frnp(x_test)]


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
    hparams = [l2_reg_params]
    # hparams: the outer variables (or hyperparameters)
    ones_dxc = torch.ones(n_features, n_classes)

    outer_opt = torch.optim.SGD(lr=args.outer_lr, momentum=args.outer_mu, params=hparams)
    # outer_opt = torch.optim.Adam(lr=0.01, params=hparams)

    params_history = []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    w = torch.zeros(n_features, n_classes).requires_grad_(True)
    parameters = [w]

    # params_history: the inner iterates (from first to last)
    if bias:
        b = torch.zeros(n_classes).requires_grad_(True)
        parameters.append(b)
 
    if args.inner_mu > 0:
        #inner_opt = hg.Momentum(train_loss, inner_lr, inner_mu, data_or_iter=train_iterator)
        inner_opt = hg.HeavyBall(train_loss, args.inner_lr, args.inner_mu, data_or_iter=train_iterator)
    else:
        inner_opt = hg.GradientDescent(train_loss, args.inner_lr, data_or_iter=train_iterator)
    inner_opt_cg = hg.GradientDescent(train_loss, 1., data_or_iter=train_iterator)

    total_time = 0
    loss_acc_time_results = np.zeros((args.n_steps+1, 3))
    test_loss, test_acc = eval(parameters, x_test, y_test)
    loss_acc_time_results[0, 0] = test_loss
    loss_acc_time_results[0, 1] = test_acc
    loss_acc_time_results[0, 2] = 0.0
    
    for o_step in range(args.n_steps):
        train_index_list = torch.randperm(train_list_len) # only minibatch need
        val_index_list = torch.randperm(val_list_len) # only minibatch need
        start_time = time.time()
        if args.alg == 'minibatch':
            inner_losses = []
            for t in range(args.T):
                loss_train = train_loss(parameters, hparams, train_list[train_index_list[t%train_list_len]])
                inner_grad = torch.autograd.grad(loss_train, parameters)
                parameters[0] = parameters[0] - args.inner_lr*inner_grad[0]
                inner_losses.append(loss_train)

                if t % train_log_interval == 0 or t == args.T-1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))

            # Gy_gradient
            Gy_gradient = gradient_gy(args, parameters, val_list[val_index_list[0]], hparams)
            print('Gy_gradient:', Gy_gradient.size())
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            # print('parameters:', parameters[0].size())
            # print('hparams:', hparams[0].size())
            
            G_gradient = torch.reshape(parameters[0], [-1]) - args.eta*Gy_gradient

            # Fy_gradient
            Fy_gradient = gradient_fy(args, parameters, val_list[val_index_list[1%val_list_len]])
            v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

            # Hessian
            z_list = []
            v_Q = args.eta*v_0
            for q in range(args.K):
                Jacobian = torch.matmul(G_gradient, v_0)
                v_new = torch.autograd.grad(Jacobian, parameters, retain_graph=True)[0]
                v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
                z_list.append(v_0)
            v_Q = v_Q+torch.sum(torch.stack(z_list), dim=0)
            print('v_Q:',v_Q.size())
            # Gyx_gradient
            Gy_gradient = gradient_gy(args, parameters, val_list[val_index_list[2%val_list_len]], hparams)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), hparams)[0]
            hparams[0] = hparams[0] - args.outer_lr*(-Gyx_gradient)
            final_params = parameters
            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)
            val_loss(final_params, hparams)

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
            if len(hparams) == 2:
                print('          l1_hp : ', torch.norm(hparams[1]))

    file_name = 'results.npy'
    file_addr = os.path.join(args.save_folder, file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_acc_time_results)   

    print(loss_acc_time_results)
    np.savetxt('stocBiO.txt',loss_acc_time_results)
    print('HPO ended in {:.2e} seconds\n'.format(total_time))

    # plt.title('val_accuracy_'+str(args.alg))
    # plt.plot(loss_acc_time_results[1:, 2], val_accs)
    # plt.show()

    # plt.title('test_accuracy_'+str(args.alg))
    # plt.plot(loss_acc_time_results[1:, 2], test_accs)
    # plt.show()

    # # Final Train on both train and validation sets
    # x_train_val = torch.cat([x_train, x_val], dim=0)
    # y_train_val = torch.cat([y_train, y_val], dim=0)
    # train_val_batch_size = len(x_train_val)


    # if train_val_batch_size < len(y_train_val):
    #     print('making iterator with batch size ', bs)
    #     train_val_iterator = CustomTensorIterator([x_train_val, y_train_val], batch_size=train_val_batch_size, shuffle=True, **kwargs)
    # else:
    #     train_val_iterator = repeat([x_train_val, y_train_val])

    # if args.inner_mu > 0:
    #     # inner_opt = hg.Momentum(train_loss, inner_lr, inner_mu, data_or_iter=train_iterator)
    #     inner_opt = hg.HeavyBall(train_loss, args.inner_lr, args.inner_mu, data_or_iter=train_val_iterator)
    # else:
    #     inner_opt = hg.GradientDescent(train_loss, args.inner_lr, data_or_iter=train_val_iterator)


    # T_final = 4000
    # w = torch.zeros(n_features, n_classes).requires_grad_(True)
    # parameters = [w]

    # if bias:
    #     b = torch.zeros(n_classes).requires_grad_(True)
    #     parameters.append(b)

    # opt_params = inner_opt.get_opt_params(parameters)

    # print('Final training on both train and validation sets with final hyperparameters')
    # for t in range(T_final):
    #     opt_params = inner_opt(opt_params, hparams, create_graph=False)
    #     train_loss = inner_opt.curr_loss

    #     if t % train_log_interval == 0 or t == T_final-1:
    #         test_loss, test_acc = eval(opt_params[:len(parameters)], x_test, y_test)
    #         print('t={} final loss: {}'.format(t, train_loss))
    #         print('          Test loss: {}, Test Acc: {}'.format(test_loss, test_acc))


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


def main():
    args = parse_args()
    print(args)
    train_model(args)

if __name__ == '__main__':
    main()
