#encoding=utf-8

import argparse
import time
from Tree import Str2Srl
from Tree import Srl2Tree

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-src_path", default='./Data/50_files.src', type=str)
    parser.add_argument("-gold_path", default='./Data/50_files.gold', type=str)
    parser.add_argument("-cand_path", default='./Data/50_files.cand', type=str)
    parser.add_argument("-srl_path", default='./Data/50_files.srl', type=str)
    parser.add_argument("-tree_path", default='./Data/50_files.tree', type=str)
    # srl parameters
    parser.add_argument('-srl_batch_size', default=30, type=int)
    parser.add_argument('-srl_archive_path', default='./Tree/srl-model-2018.05.25.tar.gz', type=str)

    args = parser.parse_args()
    srl_obj = Str2Srl.Str2Srl(args.srl_archive_path)
    tree_obj = Srl2Tree.Srl2Tree()

    srl_res = srl_obj.process(args.src_path, args.gold_path, args.cand_path)
    tree_res = tree_obj.process(srl_res)

    fpout = open(args.tree_path, 'w')
    for item in tree_res:
        fpout.write(item + '\n')
    fpout.close()
