# coding=utf8
import sys
import time

from config_default import *
from utils import Task

class Trainer(object):
    def __init__(self, task):
        self.task = task

        self.N_EPOCHS = task.config['train']['n_epochs']
        self.model = task.model
        self.optimizer = task.optimizer
        self.loss_func = task.loss_func
        self.epoch = task.epoch
        self.train_loader = task.train_loader
        self.valid_loader = task.valid_loader
        self.PRINT_EVERY = task.config['train']['print_every']
        
    def print(self, *outs):
        """ flush print """
        print(*outs)
        sys.stdout.flush()
    
    def train(self):
        self.print('Training Start ...')
        self.model.train()
        start_time = time.time()
        for epoch in range(self.epoch, self.N_EPOCHS):
            self.train_loader.shuffle() # 更换batch顺序
            total_loss = 0
            last_time = time.time()
            self.print(f'\nEpoch {epoch+1:5d}/{self.N_EPOCHS:5d}:')
            for i, batch in enumerate(self.train_loader):
                src_var, tgt_var, src_lens, tgt_lens = batch

                self.model.zero_grad()
                outputs, hidden = self.model(src_var, tgt_var[:-1], src_lens)

                loss = self.loss_func(outputs, tgt_var[1:].contiguous(), tgt_lens)
                total_loss += loss.data[0]
                self.optimizer.step()
                exit()

                if (i + 1) % self.PRINT_EVERY == 0:
                    mean_ppl = utils.PPL(total_loss / print_every)
                    self.print(f'\tBatch {i+1:5d}/{self.N_EPOCHS:5d}; Train PPL: {mean_ppl: 6.2f}; {time.time() - last_time:6.0f} s elapsed')
                    total_loss = 0
                    last_time = time.time()
                 
            self.validate()

    def validate():
        pass


def main():
    task = Task(config)
    if len(sys.argv) > 2:
        # load checkpoint
        task.load(mode='train', model_path=sys.argv[1])
    else:
        # 重新训练
        if config['train']['silence']:
            # backgrounder
            sys.stdout = open('train.log', 'w')
        task.load(mode='train')
    trainer = Trainer(task)
    trainer.train()

if __name__ == '__main__':
    main()
