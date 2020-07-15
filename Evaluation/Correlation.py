#coding=utf8

import json
import sys
import os
import numpy as np
#from evaluation_metrics import *
from scipy.stats import pearsonr
from scipy import spatial

class Correlation():
    def _label_classify(self, item):
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

    def _g2g_token_replace(self, tokens):
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

    def _filter_attn(self, article_lst, attn, task):
        phrase_attn = []
        for i, token in enumerate(article_lst):
            token_type = self._label_classify(token)
            if token_type == task:
                phrase_attn.append(attn[i])
        return phrase_attn

    def _prediction_phrase(self, article_lst, attn_dists, decoded_lst):
        ret_scores = {}; type_stuck = []; fact_stuck = []
        for i, tok in enumerate(decoded_lst):
            cls = self._label_classify(tok)
            if cls == "fact":
                type_stuck.append(cls)
                fact_stuck.append(tok)
            elif cls == "phrase":
                type_stuck.append(cls)
                tag = fact_stuck[-1][1:] + "|||" + tok[1:]
                ret_scores[tag] = self._filter_attn(article_lst, attn_dists[i], "phrase")
            elif cls == "end":
                if type_stuck.pop() == "fact":
                    fact_stuck.pop()
        return ret_scores

    def _prediction_fact(self, article_lst, attn_dists, decoded_lst):
        ret_scores = {}
        for i, tok in enumerate(decoded_lst):
            cls = self._label_classify(tok)
            if cls == "fact":
                tag = tok[1:]
                ret_scores[tag] = self._filter_attn(article_lst, attn_dists[i], "fact")
        return ret_scores

    def _get_prediction(self, attn_dists, article_lst, decoded_lst, task):
        if task == "phrase":
            return self._prediction_phrase(article_lst, attn_dists, decoded_lst)
        elif task == "fact":
            return self._prediction_fact(article_lst, attn_dists, decoded_lst)
        else:
            return self._prediction_token(article_lst, attn_dists, decoded_lst) #TODO

    def _true_false(self, g, p):
        return 1 - spatial.distance.cosine(g > 0 , p > 0.2)

    def _correlation(self, gtruth, pred, doc_id):
        corrs = []
        p_wight = []; g_weight = []
        for item in pred:
            p_wight.append(np.array(pred[item]))
        if len(pred) == 0 or len(gtruth) == 0:
            return -2, -2
        pred = sum(p_wight) / len(pred)
        for item in gtruth:
            g_weight.append(np.array(gtruth[item]))
        gtruth = sum(g_weight) / len(gtruth)
        if sum(pred) == 0:
            return -2, -2
        if sum(gtruth) == 0:
            return -2, -2
        #return pearsonr(pred, gtruth)[0], self._true_false(pred, gtruth)
        pred = pred.tolist()
        gtruth = gtruth.tolist()
        if len(pred) < 2 or len(gtruth) < 2:
            return -2, -2
        return pearsonr(pred, gtruth)[0], 0.0

    def load_cw(self, cw_path, task):
        preds = {}
        for line in open(cw_path):
            json_obj = json.loads(line.strip())
            doc_id = json_obj['doc_id']
            article_lst = json_obj['article_lst']
            decoded_lst = json_obj['decoded_lst']
            abstract_str = json_obj['abstract_str']
            attn_dists = json_obj['attn_dists']
            p_gens = json_obj['p_gens']
            #doc = self._g2g_token_replace(article_lst)
            pred = self._get_prediction(attn_dists, article_lst, decoded_lst, task)
            preds[doc_id] = pred
        return preds

    def load_human(self, gold_highlight_path, doc_trees, task):
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

    def evaluation(self, gold_highlight, pred_highlight):
        corr_all = []; corrs_01 = []
        for doc_id in gold_highlight:
            gtruth = gold_highlight[doc_id]
            pred = pred_highlight[doc_id]
            corr, corr_01 = self._correlation(gtruth, pred, doc_id)
            if corr > -2:
                corr_all.append(corr)
                corrs_01.append(corr_01)
        #print (len(corr_all))
        return sum(corr_all)/len(corr_all), sum(corrs_01)/len(corrs_01)

if __name__ == '__main__':
    # Load ground truth
    if sys.argv[1] == "human":
        doc_trees = load_doc_trees("50_trees/")
        gold_phrase_path = "AMT_data/alignment_phrase.jsonl"
        gold_highlight_phrase = load_gold(gold_phrase_path, doc_trees, "phrase")
        gold_fact_path = "AMT_data/alignment_fact.jsonl"
        gold_highlight_fact = load_gold(gold_fact_path, doc_trees, "fact")
    elif sys.argv[1] == "system":
        prediction_path = "Bert_highlight/"
        gold_highlight_phrase = load_auto_alg(prediction_path, "phrase")
        gold_highlight_fact = load_auto_alg(prediction_path, "fact")
    elif sys.argv[1] == "auto_full":
        prediction_path = "/scratch/xxu/system_trees/" + sys.argv[2] + "_gold.alg"
        gold_highlight_phrase = load_auto_alg_simple_format(prediction_path, "phrase")
        gold_highlight_fact = load_auto_alg_simple_format(prediction_path, "fact")

    # Load system alignment
    if sys.argv[1] == "auto_full":
        prediction_path = "/scratch/xxu/system_trees/" + sys.argv[2] + "_full.alg"
        highlight_phrase = load_auto_alg_simple_format(prediction_path, "phrase")
        highlight_fact = load_auto_alg_simple_format(prediction_path, "fact")
    else:
        prediction_path = "system_trees/" + sys.argv[2]
        highlight_phrase = load_auto_alg(prediction_path, "phrase")
        highlight_fact = load_auto_alg(prediction_path, "fact")
        
    # Calculate correlation
    fact_merge, fact_01 = evaluation(gold_highlight_fact, highlight_fact)
    phrase_merge, phrase_01 = evaluation(gold_highlight_phrase, highlight_phrase)

    #print ("fact_single, fact_merge ", fact_single, fact_merge)
    #print ("phrase_single, phrase_merge ", phrase_single, phrase_merge)
    print ("fact", fact_merge, fact_01)
    print ("phrase", phrase_merge, phrase_01)
