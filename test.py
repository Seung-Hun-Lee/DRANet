from __future__ import print_function
from args import get_args
from trainer import Trainer

if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)
    trainer.test()
