#coding=utf8

import json
import sys, os

def summary_process(summ):
    toks = [item for item in summ.split() if item != "<strong>" and item != "</strong>"]
    return toks

def fact_alignment(highlight_path):
    for line in open(highlight_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        uni_gram_scores = json_obj["uni_gram_scores"]
        summary_tag = json_obj["summary"]["name"]
        highlighted_summary = json_obj["summary"]["text"]
        summary = summary_process(highlighted_summary)
        summary.insert(0, "[SUMMARY]")

        out_json = {}
        out_json["article_lst"] = json_obj["document"]
        out_json["abstract_str"] = highlighted_summary
        out_json["decoded_lst"] = summary
        out_json["attn_dists"] = [uni_gram_scores]
        for i in range(len(summary)-1):
            out_json["attn_dists"].append([0.0] * len(out_json["article_lst"]))
        fpout = open("visual/" + doc_id + "-" + summary_tag + ".jsonl", "w")
        fpout.write(json.dumps(out_json))
        fpout.close()

def fact_alignment_merge(highlight_path):
    for line in open(highlight_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        uni_gram_scores = json_obj["uni_gram_scores"]
        summary_tag = json_obj["summary"]["name"]
        highlighted_summary = json_obj["summary"]["text"]
        summary = summary_process(highlighted_summary)
        summary.insert(0, "[SUMMARY]")

        out_json = {}
        out_json["article_lst"] = json_obj["document"]
        out_json["abstract_str"] = highlighted_summary
        out_json["decoded_lst"] = summary
        out_json["attn_dists"] = [uni_gram_scores]
        for i in range(len(summary)-1):
            out_json["attn_dists"].append([0.0] * len(out_json["article_lst"]))
        fpout = open("visual/" + doc_id + "-" + summary_tag + ".jsonl", "w")
        fpout.write(json.dumps(out_json))
        fpout.close()

if __name__ == '__main__':
    summary_path = "../../backend/BBC_pair/summaries/"
    highlight_path = "alignment.jsonl"
    fact_alignment(highlight_path)
