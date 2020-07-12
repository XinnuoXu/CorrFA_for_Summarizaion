#%%
"""
Extract results from database and then perform analysis on top of to draw insights
"""
import json
import os, sys
from nltk.util import ngrams
from nltk.stem.porter import *
from collections import Counter
from itertools import chain
import pandas as pd
from flask_sqlalchemy import SQLAlchemy

sys.path.append(os.path.abspath('../../'))
from backend.models import Dataset, Document, Summary, SummaryGroup, AnnotationResult, DocStatus
from backend.app import create_app

stemmer = PorterStemmer()

#%%
# Loading data from database
# summary_name = 'BBC_system_ptgen'
# summary_name = 'BBC_system_tconvs2s'
summary_name = 'BBC_ref_gold'
app = create_app()
db = SQLAlchemy(app)
results_dir = '/home/acp16hh/Projects/Research/Experiments/Exp_Elly_Human_Evaluation/results'
q_results = db.session.query(Summary, SummaryGroup, Document, Dataset) \
    .join(Document).join(SummaryGroup).join(Dataset) \
    .filter(Dataset.name == 'BBC', SummaryGroup.name == summary_name).all()


#%%
# Process data from database into components and components' type
def parse(doc_json):
    """
    Parse document into components (list of all tokens) and comp_types (list of types for all tokens)
    """
    components = []
    comp_types = []
    index = []
    for sent in doc_json['sentences']:
        for idx, token in enumerate(sent['tokens']):
            aWord = token['word'].lower()
            if token['word'] == '-LRB-':
                aWord = '('
            elif token['word'] == '-RRB-':
                aWord = ')'
            elif token['word'] == '``':
                aWord = '"'
            elif token['word'] == '\'\'':
                aWord = '"'
            components.append(aWord)
            if aWord.strip() == '':
                comp_types.append('whitespace')
            else:
                comp_types.append('word')
            index.append(len(index))
            if idx != len(sent['tokens']) - 2:
                components.append(' ')
                comp_types.append('whitespace')
                index.append(len(index))
    data = {
        'content': pd.Series(components),
        'type': pd.Series(comp_types),
        'index': pd.Series(index),
    }
    return pd.DataFrame(data)

# Contains information of each word in the document
df_doc_prop = pd.DataFrame([])

summaries = {}
for summ, _, doc, _ in q_results:
    doc_json = json.loads(doc.doc_json)
    df_doc_prop = df_doc_prop.append(parse(doc_json).assign(doc_id=doc_json['doc_id']))
    summaries[doc.doc_id] = summ.text.split()

# Contains data of the document with the summary
df_doc = pd.DataFrame(df_doc_prop[df_doc_prop['type'] != 'whitespace'].groupby('doc_id').count())
df_doc = df_doc.rename(columns={'index': 'word_len'}).drop(columns=['type', 'content'])
df_doc['summ'] = df_doc.index.map(summaries.get)

#%%
# Retrieve highlights
def process_doc(doc_json, word_idx):
    """
    Build indexes and texts for the given document
    """
    indexes = []
    texts = []
    result_ids = []
    results = doc_json['results']
    doc_id = doc_json['doc_id']
    for result_id, data in results.items():
        for h_id, highlight in data['highlights'].items():
            if highlight['text'] == '':
                continue
            word_only_highlight = [idx for idx in highlight['indexes'] if word_idx.loc[idx]['type'] == 'word']
            indexes.append(word_only_highlight)
            texts.append(highlight['text'].lower())
            result_ids.append(result_id)
    data = {
        'indexes': pd.Series(indexes),
        'text': pd.Series(texts),
        'result_id': pd.Series(result_ids),
        'doc_id': doc_id
    }
    return pd.DataFrame(data)

df_h = pd.DataFrame([])
for summ, _, doc, _ in q_results:
    doc_json = json.loads(doc.doc_json)
    word_idx = df_doc_prop.groupby('doc_id').get_group(doc_json['doc_id'])
    df_h = df_h.append(process_doc(doc_json, word_idx))

#%%
# Retrieve result per coder and store them in DataFrame
q_results = db.session.query(Document).all()

df_annotations = pd.DataFrame([])
df_doc_prop_group = df_doc_prop.groupby('doc_id')
count = 0
test = {}
for doc in q_results:
    results = json.loads(doc.doc_json)['results']
    for result_id, result in results.items():
        highlights = result['highlights']
        indexes = []
        texts = []
        word_idx = df_doc_prop_group.get_group(doc.doc_id)
        for key, highlight in highlights.items():
            if highlight['text'] == '':
                continue
            word_only_highlight = [idx for idx in highlight['indexes'] if word_idx.loc[idx]['type'] == 'word']
            indexes.append(word_only_highlight)
            texts.append(highlight['text'])
        df_annotations = df_annotations.append(
            pd.DataFrame({
                'indexes': pd.Series(indexes),
                'texts': pd.Series(texts)
            }).assign(doc_id=doc.doc_id, result_id=result_id))
# #%%
# # N_Grams
# from collections import Counter
#
#
# def calc_n_gram(x_texts, summ_text, n):
#     summ_count_gram = 0
#     summ_count_gram_match = 0
#
#     summ_unigram = list(ngrams(summ_text.split(), n))
#     # print(summ_unigram)
#     for x_text in x_texts:
#         gram = list(chain(*[list(ngrams([w.lower() for w in t.split()], n)) for t in x_text]))
#         count_gram = Counter(gram)
#         summ_count_gram += sum(count_gram.values())
#         gram_match = [gram for gram in summ_unigram if gram in count_gram.keys()]
#         count_gram_match = Counter(gram_match)
#         summ_count_gram_match += sum(count_gram_match.values())
#         # print(gram)
#         # print(sum(count_gram_match.values()) / sum(count_gram.values()))
#     return summ_count_gram_match / summ_count_gram
#
# df_ngrams = pd.DataFrame([])
# # summary_name = 'BBC_system_ptgen'
# # summary_name = 'BBC_system_tconvs2s'
# summary_name = 'BBC_ref_gold'
#
# rouge_1 = []
# rouge_2 = []
# rouge_3 = []
# data = []
# for doc_id, data in df_annotations.groupby('doc_id'):
#     summ = db.session.query(Summary, SummaryGroup, Document) \
#         .join(Document).join(SummaryGroup) \
#         .filter(
#             Dataset.name == 'BBC',
#             SummaryGroup.name == summary_name,
#             Document.doc_id == doc_id) \
#         .first()[0]
#
#     doc_idxs = list(df_doc.loc[doc_id]['doc_idxs'])
#     df_result = data.groupby('result_id')
#     x_texts = [list(chain(result['texts'])) for _, result in df_result]
#     rouge_1.append(calc_n_gram(x_texts, summ.text, 1))
#     rouge_2.append(calc_n_gram(x_texts, summ.text, 2))
#     rouge_3.append(calc_n_gram(x_texts, summ.text, 3))
# data = {
#     'rouge_1': pd.Series(rouge_1),
#     'rouge_2': pd.Series(rouge_2),
#     'rouge_3': pd.Series(rouge_3)
# }
# df_rouge = pd.DataFrame(data)
# df_rouge.describe().to_csv(os.path.join(results_dir, '%s_rouge.csv' % summary_name))

#%%
# Modified n-gram
df_doc = df_doc.assign(doc_text=lambda x: df_doc_prop[df_doc_prop['type'] == 'word'].groupby('doc_id')['content'].apply(list))
df_doc = df_doc.assign(doc_text_join=lambda x: df_doc['doc_text'].apply(' '.join))
# df_doc = df_doc.assign(h_text_join=lambda x: df_doc['h_text'].apply(' '.join))
df_doc = df_doc.assign(h_idxs=lambda x: df_h.groupby('doc_id').apply(lambda x: list(set(chain(*x.indexes)))))

df_doc = df_doc.assign(doc_idxs=lambda x: df_doc_prop[df_doc_prop['type']=='word'].groupby('doc_id')['index'].apply(list))

df_doc = df_doc.assign(h_len=lambda x: df_doc[['h_idxs']].apply(lambda x: len(x['h_idxs']), axis=1))

MAX_LEN = 30
df_ngrams = pd.DataFrame([])

#%%
def numH(w, H):
    result = 0
    H_group = H.groupby('result_id')
    highlights = {}
    for result_id, data in H_group:
        if result_id not in highlights.keys():
            highlights[result_id] = data['indexes']
        else:
            highlights[result_id].append(data['indexes'])
    for result_id in highlights.keys():
        h_words = list(chain(*[highlight for highlight in highlights[result_id]]))
        if w in h_words:
            result += len(h_words) / MAX_LEN
    return result


def beta(n, g, w, H):
    numerator = 0
    denominator = 0
    m = len(w[0])
    for i in range(m-n+1):
        total_NumH = 0
        for j in range(i, i+n):
            if w[0][i:i+n] == list(g):
                total_NumH += numH(w[1][j], H)
        total_NumH /= 10
        total_NumH /= n
        numerator += total_NumH
    for i in range(m-n+1):
        if w[0][i:i+n] == list(g):
            denominator += 1
    if denominator == 0 or numerator == 0:
        return 0
    return numerator/denominator


def R_rec(n, S, D, H):
    # stemD = [stemmer.stem(d) for d in D[0]]
    # stemS = [stemmer.stem(s) for s in S]
    n_gram_D = list(ngrams(D[0], n))
    count_n_gram_D = Counter(n_gram_D)
    n_gram_S = list(ngrams(S, n))
    count_n_gram_S = Counter(n_gram_S)

    n_gram_DnS = set(n_gram_S).intersection(set(n_gram_D))
    numerator = 0
    for g in n_gram_DnS:
        # numerator += 1 * min(count_n_gram_D[g], count_n_gram_S[g])
            numerator += beta(n, g, D, H) * min(count_n_gram_D[g], count_n_gram_S[g])
    denominator = 0
    for g in set(n_gram_D):
        denominator += beta(n, g, D, H) * count_n_gram_D[g]
        # denominator += 1 * count_n_gram_D[g]
    return numerator/max(denominator, 1)


def R_prec(n, S, D, H):
    # stemD = [stemmer.stem(d) for d in D[0]]
    # stemS = [stemmer.stem(s) for s in S]
    n_gram_D = list(ngrams(D[0], n))
    count_n_gram_D = Counter(n_gram_D)
    n_gram_S = list(ngrams(S, n))
    count_n_gram_S = Counter(n_gram_S)
    n_gram_DnS = set(n_gram_S).intersection(set(n_gram_D))
    numerator = 0
    for g in n_gram_DnS:
        # numerator += 1 * min(count_n_gram_D[g], count_n_gram_S[g])
            numerator += beta(n, g, D, H) * min(count_n_gram_D[g], count_n_gram_S[g])
    denominator = 0
    for g in set(n_gram_S):
        # denominator += beta(n, g, D, H) * count_n_gram_S[g]
        denominator += 1 * count_n_gram_S[g]
    return numerator/max(denominator, 1)


df_h_g = df_h.groupby('doc_id')
recs_1 = []
recs_2 = []
precs_1 = []
precs_2 = []
f_1s_1 = []
f_1s_2 = []
doc_ids = []
for doc_id, data in df_annotations.groupby('doc_id'):
    print(doc_id)
    summ = db.session.query(Summary, SummaryGroup, Document) \
        .join(Document).join(SummaryGroup) \
        .filter(
        Dataset.name == 'BBC',
        SummaryGroup.name == summary_name,
        Document.doc_id == doc_id) \
        .first()[0]
    doc_texts = (list(df_doc.loc[doc_id]['doc_text']), list(df_doc.loc[doc_id]['doc_idxs']))
    H = df_h_g.get_group(doc_id)
    import math
    print('Calculating 1-gram')
    r_1 = R_rec(1, summ.text.split(), doc_texts, H)
    p_1 = R_prec(1, summ.text.split(), doc_texts, H)
    if r_1 + p_1 == 0:
        f_1_1 = 0
    else:
        f_1_1 = 2 * r_1 * p_1 / (r_1 + p_1)
    print('Calculating 2-gram')
    r_2 = R_rec(2, summ.text.split(), doc_texts, H)
    p_2 = R_prec(2, summ.text.split(), doc_texts, H)
    if r_2 + p_2 == 0:
        f_1_2 = 0
    else:
        f_1_2 = 2 * r_2 * p_2 / (r_2 + p_2)
    recs_1.append(r_1)
    recs_2.append(r_2)
    precs_1.append(p_1)
    precs_2.append(p_2)
    f_1s_1.append(f_1_1)
    f_1s_2.append(f_1_2)
    doc_ids.append(doc_id)
df_f_1 = pd.DataFrame({
    'doc_id': pd.Series(doc_ids),
    'recalls_1': pd.Series(recs_1),
    'precisions_1': pd.Series(precs_1),
    'f_1s_1': pd.Series(f_1s_1),
    'recalls_2': pd.Series(recs_2),
    'precisions_2': pd.Series(precs_2),
    'f_1s_2': pd.Series(f_1s_2)
})

#%%
# Save to file
df_f_1.to_csv(os.path.join(results_dir, '%s_rouge.csv' % summary_name))
df_f_1.describe().to_csv(os.path.join(results_dir, '%s_rouge_describe.csv' % summary_name))

#%%
# Calculate word overlap ratio between document and highlights
# from itertools import chain
#
#
#
# df_doc = df_doc.assign(
#     doc_h_overlap=lambda x: df_doc[['h_idxs', 'doc_idxs']]
#         .apply(lambda x: set(x['doc_idxs']) & set(x['h_idxs']), axis=1)
#         .apply(lambda x: len(list(x))))
#
# df_doc = df_doc.assign(upper_percent=lambda x: df_doc[['h_idxs', 'doc_idxs']].apply(lambda x: len([i for i in x['h_idxs'] if i >= math.floor(max(x['doc_idxs'])/2)]) / len(x['h_idxs']), axis=1))
#
# df_doc = df_doc.assign(
#     doc_h_overlap_recall=lambda x: df_doc.doc_h_overlap/df_doc.word_len)
#
# df_doc = df_doc.assign(
#     doc_h_overlap_precision=lambda x: df_doc.doc_h_overlap/df_doc.h_len)
#
# df_doc = df_doc.assign(
#     doc_h_overlap_F1=lambda x: 2*df_doc.doc_h_overlap_recall*df_doc.doc_h_overlap_precision/(df_doc.doc_h_overlap_recall+df_doc.doc_h_overlap_precision))
# #%%
# # Calculate word overlap ratio between summary and highlights
#
# df_doc = df_doc.assign(h_text=lambda x: df_h.groupby('doc_id').apply(lambda x: ' '.join(x.text).split()))
#
#
# for i in range(1, 4):
#     kwargs_1 = {
#         'summ_h_%s_gram' % i: lambda x: df_doc[['summ', 'h_text']].apply(
#         lambda x: set(ngrams(x['h_text'], i)) & set(ngrams(x['summ'], i)), axis=1).apply(lambda x: len(list(x)))
#     }
#     kwargs_2 = {
#         'summ_h_overlap_%s_gram_recall' % i: lambda x: df_doc[['h_text', 'summ_h_%s_gram' % i]].apply(lambda x: x['summ_h_%s_gram' % i] / len(list(set(ngrams(x['h_text'], i)))), axis=1)
#     }
#     kwargs_3 = {
#         'summ_h_overlap_%s_gram_precision' % i: lambda x: df_doc[['summ', 'summ_h_%s_gram' % i]].apply(
#             lambda x: x['summ_h_%s_gram' % i] / len(list(set(ngrams(x['summ'], i)))), axis=1)
#     }
#     kwargs_4 = {
#         'summ_h_overlap_%s_gram_F1' % i: lambda x: 2*df_doc['summ_h_overlap_%s_gram_precision' % i]*df_doc['summ_h_overlap_%s_gram_recall' % i]/(df_doc['summ_h_overlap_%s_gram_precision' % i]+df_doc['summ_h_overlap_%s_gram_recall' % i])
#     }
#     df_doc = df_doc.assign(**kwargs_1)
#     df_doc = df_doc.assign(**kwargs_2)
#     df_doc = df_doc.assign(**kwargs_3)
#     df_doc = df_doc.fillna(0)
#     df_doc = df_doc.assign(**kwargs_4)
#     df_doc = df_doc.fillna(0)
# #%%
# # Calculate word overlap ratio between summary and doc
# for i in range(1, 4):
#     kwargs_1 = {
#         'summ_doc_%s_gram' % i: lambda x: df_doc[['summ', 'doc_text']].apply(
#         lambda x: set(ngrams(x['doc_text'], i)) & set(ngrams(x['summ'], i)), axis=1).apply(lambda x: len(list(x)))
#     }
#     kwargs_2 = {
#         'summ_doc_overlap_%s_gram_recall' % i: lambda x: df_doc[['doc_text', 'summ_doc_%s_gram' % i]].apply(lambda x: x['summ_doc_%s_gram' % i] / len(list(set(ngrams(x['doc_text'], i)))), axis=1)
#     }
#     kwargs_3 = {
#         'summ_doc_overlap_%s_gram_precision' % i: lambda x: df_doc[['summ', 'summ_doc_%s_gram' % i]].apply(
#             lambda x: x['summ_doc_%s_gram' % i] / len(list(set(ngrams(x['summ'], i)))), axis=1)
#     }
#     kwargs_4 = {
#         'summ_doc_overlap_%s_gram_F1' % i: lambda x: 2*df_doc['summ_doc_overlap_%s_gram_precision' % i]*df_doc['summ_doc_overlap_%s_gram_recall' % i]/(df_doc['summ_doc_overlap_%s_gram_precision' % i]+df_doc['summ_doc_overlap_%s_gram_recall' % i])
#     }
#     df_doc = df_doc.assign(**kwargs_1)
#     df_doc = df_doc.assign(**kwargs_2)
#     df_doc = df_doc.assign(**kwargs_3)
#     df_doc = df_doc.fillna(0)
#     df_doc = df_doc.assign(**kwargs_4)
#     df_doc = df_doc.fillna(0)
#
# #%%
# # Saving result to csv files
# import numpy as np
# import os
#
# result_path = '/home/acp16hh/Projects/Research/Experiments/Exp_Elly_Human_Evaluation/results'
#
# df_doc.to_csv(os.path.join(result_path, '%s_df_doc.csv' % summary_name))
# df_doc['doc_text_join'].to_csv(os.path.join(result_path, '%s_df_doc_text.csv' % summary_name))
# df_doc['h_text_join'].to_csv(os.path.join(result_path, '%s_df_h_text.csv' % summary_name))
# df_result = pd.DataFrame(df_doc.describe(include=[np.number]))
# df_result.to_csv(os.path.join(result_path, '%s_df_result.csv' % summary_name))
#
#
#
# #%%
# # Create the Fleiss' kappa matrix
# df_agreement = pd.DataFrame([])
#
# result_ids = {}
# kappa_docs = {}
# for doc_id, data in df_annotations.groupby('doc_id'):
#     print('Processing %s' % doc_id)
#     doc_idxs = list(df_doc.loc[doc_id]['doc_idxs'])
#     df_result = data.groupby('result_id')
#     x_idxs = [list(chain(*result['indexes']))
#               for _, result in df_result]
#     x_texts = [list(chain(result['texts'])) for _, result in df_result]
#     # Assigning index to each unique result_id
#     count_i = 0
#     store_i = list(data['result_id'])[0]
#     result_id2idx = {}
#     result_idx2id = {}
#     for i in list(data['result_id']):
#         if i != store_i:
#             count_i += 1
#             store_i = i
#         result_id2idx[store_i] = count_i
#         result_idx2id[count_i] = store_i
#
#     result_ids[doc_id] = data.assign(idx=lambda x: data['result_id'].apply(lambda y: result_id2idx.get(y)))
#     # Start building Fleiss' Kappa
#     coder_m = pd.DataFrame([[1 if idx in c else 0 for idx in doc_idxs] for c in x_idxs]).T
#     coder_m.index = doc_idxs
#     # Calculate P_e of Fleiss' Kappa
#     cat_m = pd.DataFrame([])
#     cat_m['c_1'] = coder_m.sum(axis=1)
#     cat_m['c_0'] = (len(x_idxs) - coder_m.sum(axis=1))
#     cat_m_sum = cat_m.sum(axis=0) / (len(x_idxs) * len(doc_idxs))
#     import math
#
#     P_e = math.pow(cat_m_sum['c_1'], 2) + math.pow(cat_m_sum['c_0'], 2)
#
#     # Commented this section because Kappa for each document is the same with mean of the following approach
#     # Calc Fleiss' Kappa for each document
#     # import numpy as np
#     #
#     # pi_m = pd.DataFrame(np.zeros((len(x_idxs), len(x_idxs))))
#     # P_i = []
#     # for i in range(len(doc_idxs)):
#     #     sum_cat_ij = 0
#     #     for j in range(2):
#     #         sum_cat_ij += \
#     #             cat_m.iloc[i][j] * (cat_m.iloc[i][j] - 1)
#     #     P_i.append(1 / (len(x_idxs) * (len(x_idxs) - 1)) * sum_cat_ij)
#     # P_o = sum(P_i) / len(doc_idxs)
#     # kappa_docs[doc_id] = (P_o - P_e) / (1 - P_e)
#
#     # Calc Fleiss' Kappa for each pair of possible combination
#     from itertools import product
#     import numpy as np
#
#     cartesian = product(range(len(x_idxs)), repeat=2)
#     pi_m = pd.DataFrame(np.zeros((len(x_idxs), len(x_idxs))))
#     for idx in cartesian:
#         if idx[0] == idx[1]:
#             continue
#         new_x_idxs = []
#         new_x_idxs.append(x_idxs[idx[0]])
#         new_x_idxs.append(x_idxs[idx[1]])
#         coder_m = pd.DataFrame([[1 if idx in c else 0 for idx in doc_idxs] for c in new_x_idxs]).T
#         coder_m.index = doc_idxs
#         cat_m = pd.DataFrame([])
#         cat_m['c_1'] = coder_m.sum(axis=1)
#         cat_m['c_0'] = (len(new_x_idxs) - coder_m.sum(axis=1))
#         P_i = []
#         for i in range(len(doc_idxs)):
#             sum_cat_ij = 0
#             for j in range(2):
#                 sum_cat_ij += \
#                     cat_m.iloc[i][j] * (cat_m.iloc[i][j] - 1)
#             P_i.append(
#                 1 / (len(new_x_idxs) * (len(new_x_idxs) - 1)) * sum_cat_ij)
#         P_o = sum(P_i) / len(doc_idxs)
#         kappa = (P_o - P_e) / (1 - P_e)
#         pi_m[idx[0]][idx[1]] = kappa
#     pi_m.sum(axis=1) / (len(x_idxs) - 1)
#     kwargs = {
#         doc_id: pi_m.sum(axis=1) / (len(x_idxs) - 1),
#         '%s_len' % doc_id: pd.Series([len(x) for x in x_idxs]),
#         '%s_text' % doc_id: pd.Series([x for x in x_texts]),
#         '%s_r_id' % doc_id: pd.Series([result_idx2id.get(i) for i in range(len(x_idxs))])
#     }
#     df_agreement = df_agreement.assign(**kwargs)
#
# #%%
# # Save agreement to files
# df_agreement_describe = df_agreement.describe()
# df_agreement_describe = df_agreement_describe.append(df_doc['doc_idxs'].apply(lambda x: len(x)))
# df_agreement.to_csv(os.path.join(result_path, 'df_agreement.csv'))
# df_annotations.to_csv(os.path.join(result_path, 'df_annotations.csv'))
# df_agreement_describe.to_csv(os.path.join(result_path, 'df_agreement_desc.csv'))
# for key, value in result_ids.items():
#     save_path = os.path.join(result_path, 'result_ids')
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     value.to_csv(os.path.join(
#             save_path, '%s_result_ids.csv' % key))



