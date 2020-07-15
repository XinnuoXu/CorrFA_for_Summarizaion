#coding=utf8

if __name__ == '__main__':
    ids = [line.strip() for line in open('./50_cases/50_ids.txt')]
    src_dir = './50_cases/50_docs/'
    cand_dir = "./50_cases/system_tconvs2s/"
    gold_dir = "./50_cases/ref_gold/"
    fpout_src = open("50_files.src", "w")
    fpout_gold = open("50_files.gold", "w")
    fpout_cand = open("50_files.cand", "w")
    for fid in ids:
        with open(src_dir + fid + '.data', 'r') as file:
            src = file.read().strip()
            fpout_src.write(src + "\n")
        with open(gold_dir + fid + '.data', 'r') as file:
            gold = file.read().strip()
            fpout_gold.write(gold + "\n")
        with open(cand_dir + fid + '.data', 'r') as file:
            cand = file.read().strip()
            fpout_cand.write(cand + "\n")
    fpout_src.close()
    fpout_gold.close()
    fpout_cand.close()
