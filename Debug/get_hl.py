#coding=utf8

import sys
import os
sys.path.append(os.path.abspath('../Highlight/'))
from highlight import *
from highlight_HROUGED import *


def simple_format(tgt_file, output):
    fpout = open(output, "w")
    for i, line in enumerate(open(tgt_file)):
        fpout.write(line.strip() + "\t" + str(i) + "\n")
    fpout.close()
    return "tmp.tgt"

if __name__ == '__main__':
    tmp_src = "input.src"
    tmp_tgt = "tmp.tree"
    tmp_output = "tmp.hl"
    tmp_tgt = simple_format(tmp_tgt, "tmp.tgt")
    dataset = DataSet(tmp_src, tmp_tgt, tmp_output, thred_num=20)
    dataset.preprocess_mult()
