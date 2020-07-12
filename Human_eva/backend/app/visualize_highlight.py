#coding=utf8

import json
import sys, os

if __name__ == '__main__':
    summary_path = "../../backend/BBC_pair/summaries/"
    highlight_path = "highlight.jsonl"
    for line in open(highlight_path):
        json_obj = json.loads(line.strip())
        doc_id = json_obj["doc_id"]
        uni_gram_scores = json_obj["uni_gram_scores"]
        with open(summary_path + doc_id + ".data", 'r') as f:
            summary = f.read().strip().split()
        summary.insert(0, "[SUMMARY]")

        out_json = {}
        out_json["article_lst"] = json_obj["document"]
        out_json["abstract_str"] = " ".join(summary)
        out_json["decoded_lst"] = summary
        out_json["attn_dists"] = [uni_gram_scores]
        for i in range(len(summary)-1):
            out_json["attn_dists"].append([0.0] * len(out_json["article_lst"]))
        fpout = open("visual/" + doc_id + ".jsonl", "w")
        fpout.write(json.dumps(out_json))
        fpout.close()