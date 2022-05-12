import argparse
import os
import random


def check_dirs(dirs):
    dirs = [dirs] if type(dirs) not in [list, tuple] else dirs
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass
    return


def get_args():
    parser = argparse.ArgumentParser()

    ## Common Parameters ##
    parser.add_argument('-T', '--task', required=True, help='clf | seg')  # Classification or Segmentation
    parser.add_argument('-D','--datasets', type=str, nargs='+', required=True, help='clf: M/MM/U (MNIST/MNIST-M/USPS) '
                                                                               'seg: G/C (GTA5/Cityscapes)')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imsize', type=int, help='the height of the input image')
    parser.add_argument('--iter', type=int, default=10000000, help='total training iterations')
    parser.add_argument('--manualSeed', type=int, default=5688)
    parser.add_argument('--ex', help='Experiment name')
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--tensor_freq', type=int, help='frequency of showing results on tensorboard during training.')
    parser.add_argument('--eval_freq', type=int, help='frequency of evaluation during training.')
    parser.add_argument('--CADT', type=bool, default=False)
    parser.add_argument('--load_step', type=int, help="iteration of trained networks")

    ## Optimizers Parameters ##
    parser.add_argument('--lr_dra', type=float, default=0.001)
    parser.add_argument('--lr_clf', type=float, default=5e-4)
    parser.add_argument('--lr_seg', type=float, default=2.5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_step', type=int, default=20000)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--weight_decay_dra', type=float, default=1e-5)
    parser.add_argument('--weight_decay_task', type=float, default=5e-4)

    args = parser.parse_args()
    check_dirs(['checkpoint/' + args.task + '/' + args.ex])
    args.logfile = './checkpoint/' + args.task + '/' + args.ex + '/' + args.ex + '.log'
    if args.task == 'seg':
        args.CADT = True
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.batch is None:
        if args.task == 'clf':
            args.batch = 32
        elif args.task == 'seg':
            args.batch = 2
    if args.imsize is None:
        if args.task == 'clf':
            args.imsize = 64
        elif args.task == 'seg':
            args.imsize = 512
    if args.tensor_freq is None:
        if args.task == 'clf':
            args.tensor_freq = 1000
        elif args.task == 'seg':
            args.tensor_freq = 100
    if args.eval_freq is None:
        if args.task == 'clf':
            args.eval_freq = 10000
        elif args.task == 'seg':
            args.eval_freq = 1000
    return args
