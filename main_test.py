import argparse
import os
import torch
import torch.nn as nn
from data_loaders import cifar10, cifar100, imagenet100, svhn, mnist, fashion_mnist
from functions import create_model, BPTT_attack, BPTR_attack, get_logger
from utils import train, val,generate_id
from attacks import FGSM,PGD,GN,SEA
parser = argparse.ArgumentParser()
parser.add_argument('-j','--workers',default=8, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=128, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd',type=str,help='optimizer')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')
# model configuration
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg11',type=str,help='model')
parser.add_argument('-T','--time',default=4, type=int,metavar='N',help='snn simulation time')
# training configuration
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
# adv training configuration
parser.add_argument('-special','--special', default='l2', type=str, help='[reg, l2]')
parser.add_argument('-beta','--beta',default=5e-4, type=float,help='regulation beta')
parser.add_argument('--attack',default='', type=str,help='attack')
parser.add_argument('-eps','--eps',default=8, type=float, metavar='N',help='attack eps')
parser.add_argument('-bpmode','--bpmode',default='bptt', type=str,help='[bptt, bptr, '']')
# only PGD
parser.add_argument('-alpha','--alpha',default=2, type=float, metavar='N', help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=4, type=int, metavar='N', help='pgd attack steps')

parser.add_argument('-enc','--encoding',default='const',type=str,help='encoding')
# parser.add_argument('-atk_enc','--atk_encoding',default='rate',type=str,help='attack encoding')
parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
parser.add_argument("--TET", action="store_true", help="Use TET loss during training")
parser.add_argument("--center", action="store_true", help="Enable mean centerring for data")
parser.add_argument('-id','--id',default='cifar10-checkpoints/vgg11_hypergeometric_T4_clean.pth',type=str,help='identifier of model to be test')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    raise RuntimeError("It's better to run on a GPU.")

def main():
    global args
    log_dir = '%s-Results'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, '%s.log'%(args.id.split('/')[-1]+args.suffix)))
    logger.info(vars(args))
      

    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        train_dataset, val_dataset, znorm = cifar10(normalized = args.center)
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
        train_dataset, val_dataset, znorm = svhn(normalized = args.center)
    elif args.dataset.lower() == 'mnist':
        num_labels = 10
        train_dataset, val_dataset, znorm = mnist(normalized = args.center)
    elif args.dataset.lower() == 'cifar100':
        num_labels = 100
        train_dataset, val_dataset, znorm = cifar100(normalized = args.center)
    elif args.dataset.lower() == 'imagenet100':
        num_labels = 100
        train_dataset, val_dataset, znorm = imagenet100(normalized = args.center)

    log_dir = '%s-checkpoints'% (args.dataset)    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # seed_all(args.seed) # im commenting this because Im not sure if it does what it should
    # replacing with this random generator 
    g = torch.Generator()
    g.manual_seed(args.seed)
    test_loader = torch.utils.data.DataLoader(val_dataset,generator=g, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    logger.info('loaded data!')  

    model = create_model(args.model.lower(), args.encoding, args.time, num_labels, znorm,False)
    checkpoint = torch.load(args.id)
    logger.info('loaded saved weights!')  

    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_simulation_time(args.time,args.bpmode)
    model.to(device)
    logger.info('model moved to device!')  

    if args.attack_mode == 'bptt':
        ff = BPTT_attack
    elif args.attack_mode == 'bptr':
        ff = BPTR_attack
    else:
        ff = None
    if args.attack.lower() == 'fgsm':
        atk = FGSM(model, device, forward_function=ff, eps=args.eps / 255, T=args.time, signed=args.center)
    elif args.attack.lower() == 'sea':
        atk = SEA(model, device, forward_function=ff, eps=args.eps, T=args.time, signed=args.center)
    elif args.attack.lower() == 'pgd':
        atk = PGD(model, device, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time, signed=args.center)
    elif args.attack.lower() == 'gn':
        atk = GN(model, device, eps=args.eps / 255, signed = args.center)
    elif args.attack.lower() == 'apgd_l1':
        print(f'APGD_L1,  model encoding:{model.encoding} , attack encoding: {atkmodel.atk_encoding} , eps={args.eps}')
        class TemporalMeanWrapper(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.model = wrapped

            def forward(self, x):
                return self.model(x).mean(0)

        model1 = TemporalMeanWrapper(model)
        model1.to(device)
        atk = AutoAttack(model1, norm='L1', eps=args.eps,version='custom',attacks_to_run=['apgd-ce'],verbose=False)
    else:
        print(f'Clean, {args.encoding} model encoding')
        atk = None
    logger.info('created attack instance!')  
    logger.info('start testing!')  
    acc = val(model, test_loader, device, args.time, atk)
    print('final Test Accu: ', acc)
    logger.info(f'Final Test Accu: {acc}')

if __name__ == "__main__":
    main()
