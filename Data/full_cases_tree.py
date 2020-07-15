#coding=utf8

import sys
import json

src_tree_path = "./full_cases/{MODEL}_src.tree"
gold_tree_path = "./full_cases/{MODEL}_gold.tree"
cand_tree_path = "./full_cases/{MODEL}_cand.tree"
ouput_path = "./{MODEL}.tree"

if __name__ == '__main__':
    model_name = sys.argv[1]
    src_tree_path = src_tree_path.replace('{MODEL}', model_name)
    gold_tree_path = gold_tree_path.replace('{MODEL}', model_name)
    cand_tree_path = cand_tree_path.replace('{MODEL}', model_name)
    ouput_path = ouput_path.replace('{MODEL}', model_name)
    src_trees = [line.strip().split('\t') for line in open(src_tree_path)]
    gold_trees = [line.strip() for line in open(gold_tree_path)]
    cand_trees = [line.strip() for line in open(cand_tree_path)]
    fpout = open(ouput_path, "w")
    for i, src in enumerate(src_trees):
        out_json = {}
        out_json["doc_id"] = str(i)
        out_json["gold_tree"] = gold_trees[i]
        out_json["cand_tree"] = cand_trees[i]
        out_json["document_trees"] = src
        fpout.write(json.dumps(out_json)+"\n")
    fpout.close()
