from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg
from lib import tflib as tl
from utils.ops import mkdir
from DCMH import DCMH
import pprint


def main(cfg):
    pprint.pprint(cfg)
    mkdir(cfg.checkpoint)
    mkdir(cfg.codefolder)
    with tl.session() as sess:
        dcmh = DCMH(sess, cfg)
        dcmh.train()



if __name__ == '__main__':
    main(cfg)

