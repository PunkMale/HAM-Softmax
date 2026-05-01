import argparse
import collections
import time
import glob
import warnings
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
# import torch.multiprocessing as mp

from model import ECAPA_TDNN
from utils import *
from datasets import BaseDataset
from loss import HyperbolicAMSoftmax

save_root = 'exps'

## record time
train_start_time = datetime.datetime.now()

parser = argparse.ArgumentParser(description="ECAPA_trainer")
parser.add_argument("--save_root",  type=str,   default=save_root)
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200)
parser.add_argument('--max_epoch',  type=int,   default=80)
parser.add_argument('--batch_size', type=int,   default=64)
parser.add_argument('--n_cpu',      type=int,   default=16)
parser.add_argument('--lr',         type=float, default=0.001)
parser.add_argument("--lr_decay",   type=float, default=0.97)
parser.add_argument("--loss_type",  type=str,   default='am')
parser.add_argument("--lambda_1",   type=float, default=1)
parser.add_argument("--lambda_2",   type=float, default=1)
parser.add_argument("--dataset",    type=str,   default='v1')
parser.add_argument('--augment',    dest='augment', action='store_true')
## Model and Loss settings
parser.add_argument('--C', type=int, default=1024)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)

## Hyperbolic Loss settings
parser.add_argument('--h_C', type=float, default=1)
parser.add_argument('--h_m', type=float, default=0.2)
parser.add_argument('--h_s', type=int, default=30)

## Command
parser.add_argument('--eval', dest='eval', action='store_true')

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

## Initialization
args = parser.parse_args()
args.postfix = None # f'_m={args.m}_s={args.s}'
args = init_args(args)

logger = Logger(args.log_path)
logger.print('start time:{}'.format(train_start_time))


class Model(nn.Module):
    def __init__(self, C, n_class, m, s, lr, lr_decay, h_C, h_m, h_s, **kwargs):
        super().__init__()
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        self.speaker_loss1 = get_loss_function(args.loss_type, n_class, m, s)
        self.speaker_loss2 = HyperbolicAMSoftmax(n_class=n_class, m=h_m, s=h_s, c=h_C).cuda()
        
        self.optim = get_optimizer(self, lr)
        self.scheduler = get_scheduler(self.optim, lr_decay)

    def forward(self, data, labels):
        speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
        loss, pred, acc = self.speaker_loss.forward(speaker_embedding, labels)
        return loss, pred, acc

    def extract_embedding(self, data):
        self.speaker_encoder.forward(data.cuda(), aug=False)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    logger.print(f"{origname} is not in the model.")
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                logger.print(f"Wrong parameter length: {origname}, model: {self_state[name].size()}, loaded: {loaded_state[origname].size()}")
                continue
            self_state[name].copy_(param)


def train(model, epoch, loader):
    model.train()
    if args.lambda_1 <= 0:
        model.speaker_loss1.eval()
    if args.lambda_2 <= 0:
        model.speaker_loss2.eval()
    # Update the learning rate based on the current epcoh
    model.scheduler.step(epoch - 1)
    lr = model.optim.param_groups[0]['lr']

    top1_1, top1_2, loss_value_1, loss_value_2, num = 0, 0, 0, 0, 1e-7
    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        if args.augment:
            indices, data, aug_data, labels = batch
        else:
            indices, data, labels = batch
        progress.set_description("train")
        labels = labels.cuda()
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        if args.lambda_1 > 0:
            loss1, output1, acc1 = model.speaker_loss1.forward(speaker_embedding, labels)
        else:
            loss1 = torch.zeros(1).cuda()
            acc1 = 0
        if args.lambda_2 > 0:
            loss2, output2, acc2, _ = model.speaker_loss2.forward(speaker_embedding, labels)
        else:
            loss2 = torch.zeros(1).cuda()
            acc2 = 0

        if args.augment:
            aug_speaker_embedding = model.speaker_encoder.forward(aug_data.cuda(), aug=True)
            if args.lambda_1 > 0:
                aug_loss1, _, _ = model.speaker_loss1.forward(aug_speaker_embedding, labels)
            else:
                aug_loss1 = torch.zeros(1).cuda()
            if args.lambda_2 > 0:
                aug_loss2, _, _, _ = model.speaker_loss2.forward(aug_speaker_embedding, labels)
            else:
                aug_loss2 = torch.zeros(1).cuda()
            loss1 = loss1.mean() + aug_loss1.mean()
            loss2 = loss2.mean() + aug_loss2.mean()
        else:
            loss1 = loss1.mean()
            loss2 = loss2.mean()
        
        nloss = args.lambda_1 * loss1 + args.lambda_2 * loss2

        model.zero_grad()
        nloss.backward()
        model.optim.step()

        num += 1
        top1_1 += acc1
        top1_2 += acc2
        loss_value_1 += loss1.item()
        loss_value_2 += loss2.item()
        progress.update()
        progress.set_postfix(
            lr='{:.4}'.format(lr),
            loss1='{:.4}'.format(loss_value_1 / num),
            loss2='{:.4}'.format(loss_value_2 / num),
            acc1='{:.4}'.format(100 * top1_1 / num),
            acc2='{:.4}'.format(100 * top1_2 / num)
        )
    progress.close()

    return loss_value_1 / num, loss_value_2 / num, lr, 100 * top1_1 / num, 100 * top1_2 / num


def main(args):
    train_dataset = BaseDataset(**vars(args))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.n_cpu,
                            drop_last=True)
    args.n_class = train_dataset.get_speaker_number()
    evaluator = get_evaluator(args.dataset)
    EERs = [100]
    ## Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()
    if len(modelfiles) >= 1:
        logger.print(f"Model {modelfiles[-1]} loaded from previous state!")
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        model = Model(**vars(args))
        model.load_parameters(modelfiles[-1])
    ## Otherwise, system will train from scratch
    else:
        epoch = 1
        model = Model(**vars(args))
    if not os.path.exists(args.score_save_path):
        with open(args.score_save_path, "w") as f:
            f.write("Time,Epoch,LR,Loss1,Loss2,Acc1,Acc2,EER,minDCF(0.05),bestEER\n")
            f.close()
    
    score_file = open(args.score_save_path, "a+")

    while epoch <= args.max_epoch:
        logger.print("Epoch {}:".format(epoch))
        epoch_start_time = datetime.datetime.now()
        logger.print(f"{epoch_start_time}")

        # Training
        loss1, loss2, lr, acc1, acc2 = train(model, epoch, train_loader)

        # Save Model
        model.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)

        # Evaluation every [test_step] epochs
        EER, minDCF05 = None, None
        if epoch % args.test_step == 0:
            EER, minDCF05 = evaluator.eval(model)
            EERs.append(EER)
            logger.print(f"ACC1: {acc1:.3f}, ACC2: {acc2:.3f}, EER: {EER:.3f}, minDCF: {minDCF05:.3f}, bestEER {min(EERs):.3f}")
        score_file.write(
            "{},{},{},{},{},{},{},{},{},{}\n".format(datetime.datetime.now(), epoch, lr, loss1, loss2, acc1, acc2, EER,
                                                  minDCF05, min(EERs)))
        score_file.flush()
        epoch += 1
        ## record time
        epoch_end_time = datetime.datetime.now()
        logger.print('this epoch time:{}\n'.format(epoch_end_time - epoch_start_time))

    # 加载final模型
    final_model_path = os.path.join(args.model_save_path, "model_final.model")
    if os.path.exists(final_model_path):
        ensemble_model = Model(**vars(args))
        ensemble_model.load_parameters(final_model_path)
        logger.print(f"{final_model_path} is loaded!")
    else:
        models = []
        for i in range(5):
            model_path = os.path.join(args.model_save_path, "model_%04d.model" % (args.max_epoch - i))
            model = Model(**vars(args))
            model.load_parameters(model_path)
            logger.print("{} is loaded!".format(model_path))
            models.append(model)
        ensemble_model = Model(**vars(args))
        worker_state_dict = [x.state_dict() for x in models]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(models)):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / len(models)
        ensemble_model.load_state_dict(fed_state_dict)
        ensemble_model.save_parameters(args.model_save_path + "/model_final.model")
    
    logger.print("start to evaluate the final model")
    EER, minDCF05 = evaluator.eval(ensemble_model)
    EERs.append(EER)
    logger.print(f"final, EER: {EER:.3f}, minDCF(0.05): {minDCF05:.3f}, bestEER: {min(EERs):.3f}")
    score_file = open(args.score_save_path, "a+")
    score_file.write(
        "{},{},,,,,,{},{},{}\n".format(datetime.datetime.now(), "final", EER, minDCF05,
                                        min(EERs)))
    score_file.flush()

    if args.dataset == 'v2':
        logger.print("evaluate the final model in Vox-E and Vox-H")
        vox_eh_evaluator = get_vox_eh_evaluator()
        results = vox_eh_evaluator.eval(ensemble_model)
        with open(os.path.join(args.save_path, 'Vox-EH.csv'), "w+") as f:
            f.write("testlist,EER,minDCF(0.05)\n")

            EER, minDCF05 = results[0]
            logger.print(f"Vox-E, EER: {EER:.3f}, minDCF(0.05): {minDCF05:.3f}")
            f.write("Vox-E,{},{}\n".format(EER, minDCF05))

            EER, minDCF05 = results[1]
            logger.print(f"Vox-H, EER: {EER:.3f}, minDCF(0.05): {minDCF05:.3f}")
            f.write("Vox-H,{},{}\n".format(EER, minDCF05))

    # record time
    train_end_time = datetime.datetime.now()
    logger.print('total train time:{}\n'.format(train_end_time - train_start_time))


if __name__ == '__main__':
    # mp.set_start_method('spawn')  # 关键
    time.sleep(5)
    main(args)
