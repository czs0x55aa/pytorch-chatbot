# coding=utf8
import sys
from config_default import *

def train(task):
    pass

def main():
    if len(sys.argv) > 2:
        # load checkpoint
        pass
    else:
        # 重新训练
        if config['train']['silence']:
            # backgrounder
            sys.stdout = open('train.log', 'w')


if __name__ == '__main__':
    main()
