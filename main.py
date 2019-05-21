from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg
from lib import tflib as tl
from utils.ops import mkdir
from framework import Model
import pprint


def main(cfg):
    pprint.pprint(cfg)
    mkdir(cfg.checkpoint)
    mkdir(cfg.codefolder)
    with tl.session() as sess:
        dcmh = Model(sess, cfg)
        dcmh.train()



if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main(cfg)

