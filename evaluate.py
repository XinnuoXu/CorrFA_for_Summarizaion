#encoding=utf-8

import argparse
import time
import json
from Evaluation import Str2Srl
from Evaluation import Srl2Tree
from Evaluation import CWeighting
from Evaluation import Correlation

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_from_tree(tree_res):
    doc_trees = []; gold_trees = []; cand_trees = []
    for res in tree_res:
        res = json.loads(res)
        doc_id = res["doc_id"]
        gold_tree = res["gold_tree"] + "\t" + doc_id
        cand_tree = res["cand_tree"] + "\t" + doc_id
        doc_tree = '\t'.join(res["document_trees"])
        doc_trees.append(doc_tree)
        gold_trees.append(gold_tree)
        cand_trees.append(cand_tree)
    return doc_trees, gold_trees, cand_trees

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # srl/tree parameters
    parser.add_argument("-src_path", default='./Data/50_files.src', type=str)
    parser.add_argument("-gold_path", default='./Data/50_files.gold', type=str)
    parser.add_argument("-cand_path", default='./Data/50_files.cand', type=str)
    parser.add_argument('-srl_archive_path', default='./Evaluation/srl-model-2018.05.25.tar.gz', type=str)
    parser.add_argument('-srl_batch_size', default=30, type=int)

    # Content weighting parameters
    parser.add_argument("-cw_gold_path", default='./Data/cw_gold', type=str)
    parser.add_argument("-cw_cand_path", default='./Data/cw_cand', type=str)
    parser.add_argument('-cw_thred_num', default=20, type=int)

    # Process control
    parser.add_argument("-run_srl", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-srl_path", default='', type=str)
    parser.add_argument("-run_tree", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-tree_path", default='', type=str)

    args = parser.parse_args()

    '''
    @@ Get SRL
    '''
    if args.run_srl:
        # Get srls for doc, gold, cand
        srl_obj = Str2Srl.Str2Srl(args.srl_archive_path)
        srl_res = srl_obj.process(args.src_path, args.gold_path, args.cand_path)
    elif not args.run_tree:
        pass
    elif args.srl_path == '':
        print ('[ERROR] srl_path can\'t be empty if run_srl is False')
        exit(1)
    else:
        # Read srls from srl_path for doc, gold, cand
        # TODO
        print ("[ERROR] Still in TODO list")
        exit(0)

    '''
    @@ Get Tree
    '''
    if args.run_tree:
        # Get trees for doc, gold, cand
        tree_obj = Srl2Tree.Srl2Tree()
        tree_res = tree_obj.process(srl_res)
        doc_trees, gold_trees, cand_trees = read_from_tree(tree_res)
    elif args.tree_path == '':
        print ('[ERROR] tree_path can\'t be empty if run_tree is False')
        exit (1)
    else:
        # Read trees from tree_path for doc, gold, cand
        tree_res = [line.strip() for line in open(args.tree_path)]
        doc_trees, gold_trees, cand_trees = read_from_tree(tree_res)

    '''
    @@ Get Content Weights
    '''
    # Get Content Weight refering to Gold
    gold_set = CWeighting.DataSet(doc_trees, gold_trees, args.cw_gold_path, thred_num=args.cw_thred_num)
    gold_set.preprocess_mult()
    # Get Content Weight refering to Cand
    cand_set = CWeighting.DataSet(doc_trees, cand_trees, args.cw_cand_path, thred_num=args.cw_thred_num)
    cand_set.preprocess_mult()


    '''
    @@ Get Pearson Correlation Coefficient (PCC)
    '''
    # Get Correlation
    corr_obj = Correlation.Correlation()
    preds_gold_ph = corr_obj.load_cw(args.cw_gold_path, "phrase")
    preds_gold_fa = corr_obj.load_cw(args.cw_gold_path, "fact")

    preds_cand_ph = corr_obj.load_cw(args.cw_cand_path, "phrase")
    preds_cand_fa = corr_obj.load_cw(args.cw_cand_path, "fact")

    fact_merge, fact_01 = corr_obj.evaluation(preds_gold_fa, preds_cand_fa)
    phrase_merge, phrase_01 = corr_obj.evaluation(preds_gold_ph, preds_cand_ph)

    print ("Corr-F", fact_merge)
    print ("Corr-A", phrase_merge)
