#coding=utf8

import json
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../Highlight_evaluation/'))
from evaluation_metrics import *
from scipy.stats import pearsonr

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
    #threshold = max(np.sort(tok_align)[-1] * 0.6, np.sort(tok_align)[-n])
    threshold = np.sort(tok_align)[-1] * 0.3
    for i in range(len(tok_align)):
        if tok_align[i] < threshold:
            tok_align[i] = 0.0
    return tok_align

def _reweight_for_showing(attn):
    return (np.array(attn) * 0.8 / max(attn)).tolist()

def _fact_to_token_weight(decoded_lst, p_gens):
    fact_weights = []; lebel_stack = []; new_tokens = []; new_p_gens = []
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            lebel_stack.append(cls)
            fact_weights.append(p_gens[i])
        elif cls == "phrase":
            lebel_stack.append(cls)
        elif cls == "end":
            if lebel_stack.pop() == "fact":
                fact_weights.pop()
        elif cls == "token":
            new_tokens.append(tok)
            if len(fact_weights) > 0:
                new_p_gens.append(fact_weights[-1])
            else:
                new_p_gens.append(0.0)
    return new_tokens, new_p_gens

        
def get_p_gens(attn_dists, decoded_lst):
    p_gens = []
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            p_gens.append(max(attn_dists[i]))
        else:
            p_gens.append(0.0)
    return _fact_to_token_weight(decoded_lst, p_gens)

def get_attn_dists(attn_dists, article_lst, decoded_lst):
    fact_weights = []
    for i, tok in enumerate(decoded_lst):
        cls = _label_classify(tok)
        if cls == "fact":
            fact_weights.append(np.array(attn_dists[i]))
    fact_weights = _reweight_for_showing(sum(fact_weights).tolist())
    return  _fact_to_token_weight(article_lst, fact_weights)


def plot_data(tag):
    filename = "output/" + tag + ".txt"
    label = "[FACT WEIGHT]"

    with open(filename, 'r') as file:
        json_obj = json.loads(file.read().strip())

    decoded_lst = json_obj['decoded_lst']
    article_lst = json_obj['article_lst']
    attn_dists = json_obj['attn_dists']
    new_decoded_lst, p_gens = get_p_gens(attn_dists, decoded_lst)
    new_article_lst, token_weights = get_attn_dists(attn_dists, article_lst, decoded_lst)
    new_attn_dists = [[0] * len(new_article_lst) for i in range(len(new_decoded_lst))]
    new_attn_dists.insert(0, token_weights)
    new_decoded_lst.insert(0, label)

    json_obj['article_lst'] = new_article_lst
    json_obj['decoded_lst'] = new_decoded_lst
    json_obj['p_gens'] = [0] * len(new_decoded_lst)
    json_obj['p_gens'][0] = 0.8
    json_obj['attn_dists'] = new_attn_dists
    json_obj["abstract_str"] = "..."

    fpout = open("output/" + tag + ".hl", "w")
    fpout.write(json.dumps(json_obj) + "\n")
    fpout.close()

if __name__ == '__main__':
    for filename in os.listdir("./output"):
        if filename.endswith('.txt') and filename != "corr.txt":
            tag = filename.replace(".txt", "")
            plot_data(tag)
