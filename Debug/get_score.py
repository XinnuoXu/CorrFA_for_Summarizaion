#coding=utf8

import json
import sys
import os
import numpy as np
from scipy.stats import pearsonr
from scipy import spatial
sys.path.append(os.path.abspath('../Highlight_evaluation/'))
from alignment_check import get_ground_truth
from evaluation_metrics import *

def _label_classify(item):
    if item[0] == '(':
        if item[1] == 'F':
            return "fact"
        else:
            return "phrase"
    elif item[0] == ')':
        return "end"
    elif item[0] == '*':
        return "reference"
    return "token"

def _top_n_filter(tok_align, n):
    if n == -1:
        return tok_align
    threshold = np.sort(tok_align)[-n]
    for i in range(len(tok_align)):
        if tok_align[i] < threshold:
            tok_align[i] = 0.0
    return tok_align

def _g2g_token_replace(tokens):
    at_at_num = 0
    for i in range(len(tokens)):
        if tokens[i] == "@@":
            at_at_num += 1
            if at_at_num % 2 == 1:
                tokens[i] = "("
            else:
                tokens[i] = ")"
        elif tokens[i] == "£":
            tokens[i] = "#"
        elif tokens[i] == "[":
            tokens[i] = "-lsb-"
        elif tokens[i] == "]":
            tokens[i] = "-rsb-"
    return tokens

def load_gold(gold_highlight_path, doc_trees, task):
    gold_highlight = {}
    for line in open(gold_highlight_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        summ = json_obj["summary"]
        if isinstance(summ, dict):
            tag = summ["name"]
            if doc_id not in gold_highlight:
                gold_highlight[doc_id] = {}
            gold_highlight[doc_id][tag] = json_obj
        else:
            gold_highlight[doc_id] = json_obj

    gtruths = {}
    for doc_id in gold_highlight: 
        article_lst = doc_trees[doc_id]
        gtruth = get_ground_truth(article_lst, gold_highlight[doc_id], task)
        gtruths[doc_id] = gtruth

    return gtruths

def _filter_attn(article_lst, attn, task):
    phrase_attn = []
    for i, token in enumerate(article_lst):
        token_type = _label_classify(token)
        if token_type == task:
            phrase_attn.append(attn[i])
    return phrase_attn

def _prediction_phrase(article_lst, attn_dists, decoded_lst):
    ret_scores = {}; type_stuck = []; fact_stuck = []
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            type_stuck.append(cls)
            fact_stuck.append(tok)
        elif cls == "phrase":
            type_stuck.append(cls)
            tag = fact_stuck[-1][1:] + "|||" + tok[1:]
            ret_scores[tag] = _filter_attn(article_lst, attn_dists[i], "phrase")
        elif cls == "end":
            if type_stuck.pop() == "fact":
                fact_stuck.pop()
    return ret_scores

def _prediction_fact(article_lst, attn_dists, decoded_lst):
    ret_scores = {}
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            tag = tok[1:]
            ret_scores[tag] = _filter_attn(article_lst, attn_dists[i], "fact")
    return ret_scores

def get_prediction(attn_dists, article_lst, decoded_lst, task):
    if task == "phrase":
        return _prediction_phrase(article_lst, attn_dists, decoded_lst)
    elif task == "fact":
        return _prediction_fact(article_lst, attn_dists, decoded_lst)
    else:
        return _prediction_token(article_lst, attn_dists, decoded_lst)

def true_false(g, p):
    g_ = np.array(g)
    p_ = np.array(p)
    return 1 - spatial.distance.cosine(g_ > 0 , p_ > 0.2)

def correlation(gtruth, pred, doc_id):
    corrs = []
    p_wight = []; g_weight = []
    for item in pred:
        p_wight.append(np.array(pred[item]))
    if len(pred) == 0:
        return -2, -2
    pred = sum(p_wight) / len(pred)
    for item in gtruth:
        g_weight.append(np.array(gtruth[item]))
    gtruth = sum(g_weight) / len(gtruth)
    if sum(pred) == 0:
        return -2, -2
    if sum(gtruth) == 0:
        return -2, -2
    #return pearsonr(pred, gtruth)[0], true_false(pred, gtruth)
    return pearsonr(pred.tolist(), gtruth.tolist())[0], 0.0

def load_auto_alg_simple_format(prediction_path, task):
    preds = {};
    for line in open(prediction_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj['doc_id']

        article_lst = json_obj['article_lst']
        decoded_lst = json_obj['decoded_lst']
        abstract_str = json_obj['abstract_str']
        attn_dists = json_obj['attn_dists']
        p_gens = json_obj['p_gens']
        #doc = _g2g_token_replace(article_lst)

        pred = get_prediction(attn_dists, article_lst, decoded_lst, task)
        preds[doc_id] = pred
    return preds

def eva_debug(highlights):
    corr_all = {}
    gtruth = highlights['0']
    for doc_id in highlights:
        pred = highlights[doc_id]
        corr, corr_01 = correlation(gtruth, pred, doc_id)
        corr_all[doc_id] = corr
    return corr_all

def format_tansfer(highlight):
    new_hl = []
    for line in highlight:
        jobj = json.loads(line)
        jobj['p_gens'] = [item if item >= 0 else 0 for item in jobj['p_gens']]
        new_hl.append(json.dumps(jobj))
    return new_hl

if __name__ == '__main__':
    prediction_path = "tmp.hl"
    hl_phrase = load_auto_alg_simple_format(prediction_path, "phrase")
    hl_fact  = load_auto_alg_simple_format(prediction_path, "fact")

    # Calculate correlation
    fact_weight = eva_debug(hl_fact)
    phrase_weight = eva_debug(hl_phrase)

    # Prepare output
    os.system("rm -rf output; mkdir output")
    src = [line.strip() for line in open("input.src")]
    tgt = [line.strip() for line in open("input.tgt")]
    hl = format_tansfer([line.strip() for line in open("tmp.hl")])
    fpout_res = open("output/corr.txt", "w")
    for i, s in enumerate(src):
        idx = str(i)
        fpout_debug = open("output/" + idx + ".txt", "w")
        fpout_debug.write(hl[i])
        fpout_debug.close()
        fpout_res.write(s + "\t" + tgt[i] + "\t" + str(fact_weight[idx]) + "\t" + str(phrase_weight[idx]) + "\n")
    fpout_res.close()
    os.system("rm tmp.*")
