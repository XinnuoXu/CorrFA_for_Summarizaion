import sys
import json
#from label import get_batch_trees
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

class Str2Srl():

    def __init__(self, archive_path):
        print ("Initializing Str2Srl...")
        self.archive_path = "srl-model-2018.05.25.tar.gz"
        self.archive_path = archive_path
        self.archive = load_archive(self.archive_path, cuda_device=0)
        self.srl = Predictor.from_archive(self.archive)
        self.batch_size = 30
        print ("Initializing Str2Srl Done...")

    def get_srl(self, sentences):
        one_file = {}
        sentences = [{"sentence": line} for line in sentences]
        srl_res = []; start_idx = 0
        while start_idx < len(sentences):
            batch_sentences = sentences[start_idx: min(start_idx + self.batch_size, len(sentences))]
            srl_res.extend(self.srl.predict_batch_json(batch_sentences))
            start_idx += self.batch_size
        if len(srl_res) > 1:
            one_file["srl_document"] = srl_res[:-2]
            one_file["srl_gold"] = srl_res[-2]
            one_file["srl_cand"] = srl_res[-1]
            one_file["document"] = sentences[:-2]
            one_file["gold"] = sentences[-2]
            one_file["cand"] = sentences[-1]
            return json.dumps(one_file)
        return ""

    def process(self, src_file, gold_file, cand_file):
        print ("Building Srls...")
        srcs = [line.strip() for line in open(src_file)]
        golds = [line.strip() for line in open(gold_file)]
        cands = [line.strip() for line in open(cand_file)]
        outputs = []
        for i, line in enumerate(srcs):
            sentences = line.strip().split('\t')
            sentences.append(golds[i])
            sentences.append(cands[i])
            srl_res = self.get_srl(sentences)
            if srl_res != "":
                outputs.append(srl_res)
        print ("Building Srls Done...")
        return outputs

if __name__ == '__main__':
    src_file = "../Data/" + sys.argv[1]
    gold_file = "../Data/" + sys.argv[2]
    cand_file = "../Data/" + sys.argv[3]
    out_file = "../Data/" + sys.argv[4]
    srl_obj = Str2Srl()

    fpout = open(out_file, 'w')
    for srl_res in srl_obj.process(src_file, gold_file, cand_file):
        fpout.write(srl_res + "\n")
    fpout.close()
