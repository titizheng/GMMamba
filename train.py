

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
sys.path.append(os.path.abspath('/home/ttzheng'))


from utils.utils import make_parse
from main import train_main


if __name__ == "__main__":
    args = make_parse()
    args.flod =  "flod0"
    args.arch = 'TCGA-ESCA'
    train_main(args)
