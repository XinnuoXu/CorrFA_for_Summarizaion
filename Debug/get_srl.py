import sys
import os
import sys
import json
import os.path as path
sys.path.append('../Fact_extraction/')
from srl_tree import one_summary
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

def get_srl(sentences, output_path):
        archive = load_archive("../Fact_extraction/srl-model-2018.05.25.tar.gz", cuda_device=0)
        srl = Predictor.from_archive(archive)
        batch_size = 5
        print ("Loading Done")

        fpout = open(output_path, "w")
        sentences = [{"sentence": line} for line in sentences]
        srl_res = []; start_idx = 0
        while start_idx < len(sentences):
                batch_sentences = sentences[start_idx: min(start_idx + batch_size, len(sentences))]
                srl_res.extend(srl.predict_batch_json(batch_sentences))
                start_idx += batch_size
        for i, res in enumerate(srl_res):
                fpout.write(json.dumps(res) + "\n")
        fpout.close()

if __name__ == '__main__':
        if sys.argv[1] == "tree":
                fpout = open("tmp.tree", "w")
                for line in open("tmp.srl"):
                        fpout.write(one_summary(json.loads(line.strip())) + "\n")
                fpout.close()

        if sys.argv[1] == "srl":
                path = "input.tgt"; lines = []
                for i, line in enumerate(open(path, 'r')):
                        lines.append(line.strip())
                get_srl(lines, "tmp.srl")
