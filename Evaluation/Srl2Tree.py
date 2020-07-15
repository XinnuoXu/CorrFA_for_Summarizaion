import sys
import json
import re

class Srl2Tree():

    def one_verb(self, v_item, slen):
        if ("verb" not in v_item) \
            or ("tags" not in v_item) \
            or ("description" not in v_item) \
            or "B-V" not in v_item["tags"]:
            return -1, -1, -1, []
        b_idx = 0; e_idx = slen - 1
        # Fact begin index
        while b_idx < slen and v_item["tags"][b_idx] == "O":
            b_idx += 1
        # Fact end index
        while e_idx >= 0 and v_item["tags"][e_idx] == "O":
            e_idx -= 1
        e_idx += 1
        # Fact verb index
        v_idx = v_item["tags"].index("B-V")
        if b_idx >= e_idx:
            return -1, -1, -1, []
        # Extract semantic roles
        segments = []; seg_b_idx = -1; last_seg_label = ""
        for i in range(b_idx, e_idx):
            if v_item["tags"][i][0] == "B":
                if seg_b_idx != -1:
                    segments.append((seg_b_idx, i, last_seg_label))
                last_seg_label = v_item["tags"][i][2:]
                seg_b_idx = i
            elif v_item["tags"][i][0] == "O":
                if seg_b_idx != -1:
                    segments.append((seg_b_idx, i, last_seg_label))
                last_seg_label = ""; seg_b_idx = -1
        if seg_b_idx != -1:
            segments.append((seg_b_idx, e_idx, last_seg_label))
        return b_idx, e_idx, v_idx, segments

    def get_tree(self, facts, words, slen):
        fact_id = 1; f_labels = [""] * slen; e_labels = [""] * slen; o_labels = [""] * slen; seg_collection = {}
        for item in sorted(facts.items(), key = lambda d:d[1][0], reverse = True):
            (plen, b_idx, e_idx, v_idx, segments) = item[1]
            if b_idx < 0 or e_idx > slen:
                continue
            # Label facts
            overlap = ""; b_idx = -1
            for seg in segments:
                (seg_b, seg_e, seg_type) = seg
                if (seg_b, seg_e) not in seg_collection:
                    if b_idx == -1:
                        b_idx = seg_b
                        f_labels[b_idx] += "(F" + str(fact_id) + "-" + words[v_idx] + " "
                        e_labels[e_idx-1] += " )";
                    seg_collection[(seg_b, seg_e)] = "F" + str(fact_id) + "-" + seg_type
                    f_labels[seg_b] += overlap; overlap = ""
                    f_labels[seg_b] += "(" + seg_type + " "
                    e_labels[seg_e-1] += " )"
                else:
                    overlap += "(" + seg_type + " *trace-" + seg_collection[(seg_b, seg_e)] + "* ) "
            #e_labels[e_idx-1] += " )";
            fact_id += 1
        for i in range(0, slen):
            if words[i] == '(':
                words[i] = "@@"
            elif words[i] == ')':
                words[i] = "@@"
            words[i] = f_labels[i] + words[i] + e_labels[i]
        return words
            
    def final_tune(self, tree, slen):
        b_idx = 0; e_idx = slen - 1
        while b_idx < slen and \
            len(tree[b_idx]) == 1 and \
            not tree[b_idx][0].isalpha() and \
            not tree[b_idx][0].isdigit():
                b_idx += 1
        while e_idx >= 0 and \
            len(tree[e_idx]) == 1 and \
            not tree[e_idx][0].isalpha() and \
            not tree[e_idx][0].isdigit():
                e_idx -= 1
        tree = tree[b_idx:e_idx+1]
        if len(tree) > 0 and (not tree[0].startswith("(F")):
            tree[0] = "(F0 " + tree[0]
            tree[-1] += " )"
        return tree

    def one_summary(self, item):
        if "words" not in item or "verbs" not in item:
            return 
        words = item["words"]; slen = len(words); facts = {}; summary = " ".join(words)
        for i, v_item in enumerate(item["verbs"]):
            b_idx, e_idx, v_idx, segments = self.one_verb(v_item, slen)
            if b_idx == -1 or len(segments) == 1:
                continue
            if (b_idx, e_idx) in facts:
                b_idx = v_idx
            facts[(b_idx, e_idx)] = (e_idx - b_idx, b_idx, e_idx, v_idx, segments)
        tree = self.get_tree(facts, words, slen)
        tree = self.final_tune(tree, slen)
        return " ".join(tree)

    def process(self, inputs):
        print ("Building Trees...")
        outputs = []
        for line in inputs:
            srl_res = json.loads(line)
            gold_tree = self.one_summary(srl_res["srl_gold"])
            cand_tree = self.one_summary(srl_res["srl_cand"])
            document_trees = [self.one_summary(item) for item in srl_res["srl_document"]]
            out_json = {}
            out_json["doc_id"] = srl_res["doc_id"]
            out_json["gold_tree"] = gold_tree
            out_json["cand_tree"] = cand_tree
            out_json["document_trees"] = document_trees
            outputs.append(json.dumps(out_json))
        print ("Building Trees Done...")
        return outputs


if __name__ == '__main__':
    input_file = "../Data/" + sys.argv[1]
    inputs = [line.strip() for line in open(input_file)]

    tree_obj = Srl2Tree()
    output_file = "../Data/" + sys.argv[2]
    fpout = open(output_file, "w")
    for res in tree_obj.process(inputs):
        fpout.write(res + "\n")
    fpout.close()
