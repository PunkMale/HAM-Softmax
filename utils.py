import os
import torch
from loss import *
from tools import Logger


def get_save_path(args, method):
    save_path = os.path.join(args.save_root, f"{args.dataset}")

    save_path += f"_lam1={args.lambda_1}"
    if args.lambda_1 > 0:
        save_path += f'_{args.loss_type}'
        if 'sub' in args.loss_type:
            save_path = f"_K={args.K}"
    
    save_path += f"_lam2={args.lambda_2}"
    if args.lambda_2 > 0:
        if method == 'ham':
            if args.h_C < 1:
                save_path += f'_C={args.h_C}'
            else:
                save_path += f'_C={int(args.h_C)}'
            if args.h_m == 0:
                save_path += f'_m={0}'
            else:
                save_path += f'_m={args.h_m}'
            save_path += f'_s={args.h_s}'

    if args.augment:
        save_path += "_augment"

    if args.postfix:
        if isinstance(args.postfix, list):
            postfix = '_'.join(args.postfix)
            save_path += f"_{postfix}"
        else:
            save_path += f'_{args.postfix}'
    
    return save_path

def init_args(args, method='ham'):
    args.save_path = get_save_path(args, method)
    print("save path:", args.save_path)

    args.log_path = os.path.join(args.save_path, 'train.log')
    args.score_save_path = os.path.join(args.save_path, 'result_{}.csv'.format(args.save_path.split('/')[-1]))
    args.log_file = os.path.join(args.save_path, '{}.log'.format(args.save_path.split('/')[-1]))
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    if  args.dataset == 'v1':
        args.test_step = 10
        args.max_epoch = 150
    elif args.dataset == 'v2':
        args.test_step = 5
        args.max_epoch = 100
    elif args.dataset == 'cn':
        args.test_step = 5
        args.max_epoch = 100
    else:
        print(f"Unsupport dataset: {args.dataset}!")
        exit()
    
    args.train_list, args.train_path = get_data_path(args.dataset)
    
    return args


def get_logger(args):
    return Logger(args.log_file)


def get_loss_function(loss_type, n_class, m, s, K=3, output=False):
    if loss_type == 'ce':
        return Softmax(n_class=n_class).cuda()
    elif loss_type == 'ces':
        return SoftmaxScale(n_class=n_class).cuda()
    elif loss_type == 'am':
        return AMSoftmax(n_class=n_class, m=m, s=s).cuda()
    elif loss_type == 'aam':
        return AAMSoftmax(n_class=n_class, m=m, s=s).cuda()
    elif loss_type == 'ram':
        return RAMSoftmax(n_class=n_class, m=m, s=s).cuda()
    else:
        print("Loss type not supported!")
        exit(0)


def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)


def get_scheduler(optimizer, lr_decay):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)


def get_data_path(dataset):
    vox1_dev_path = '/home/fangzh21/data/voxceleb/voxceleb1/dev/wav/'
    vox2_dev_path = '/home/fangzh21/data/voxceleb/voxceleb2/dev/aac/'
    cn_dev_path = '/home/fangzh21/data/cnceleb'

    if 'v1' == dataset:
        train_path = vox1_dev_path
    elif 'v2' == dataset:
        train_path = vox2_dev_path
    elif 'cn' == dataset:
        train_path = cn_dev_path
    else:
        train_path = None
    train_list = f"data/{dataset}_clean.txt"
    print('train_list:', train_list)
    return train_list, train_path


def get_evaluator(dataset):
    from datasets import Evaluator, FastEvaluator

    if 'v1' in dataset or 'v2' in dataset:
        eval_list = 'data/vox_O.txt'
        eval_path = "/home/fangzh21/data/voxceleb/voxceleb1"
    elif 'cn' in dataset:
        eval_list = 'data/CN.Eval_list.txt'
        eval_list1 = '/home/fangzh21/data/cn_file/cn_veri_test.txt'
        eval_list2 = '/home/fangzh21/data/cn_file/cn_veri_test.txt'
        eval_path = '/home/fangzh21/data/cnceleb/cn_1/eval'
        return FastEvaluator(eval_list, eval_list1, eval_list2, eval_path)
    else:
        print("Dataset not supported!")
        exit(0)
    return Evaluator(eval_list, eval_path)


def get_vox_eh_evaluator():
    from datasets import FastEvaluator

    eval_path = "/home/fangzh21/data/voxceleb/voxceleb1"
    return FastEvaluator(
        'data/vox_EH_list.txt',
        'data/vox_E.txt',
        'data/vox_H.txt',
        eval_path
    )
