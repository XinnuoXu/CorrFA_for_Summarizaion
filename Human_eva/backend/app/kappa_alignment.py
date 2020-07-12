#%%
"""
Extract results from database and then perform analysis on top of to draw insights
"""
import json
import os, sys
import pandas as pd
from flask_sqlalchemy import SQLAlchemy

sys.path.append(os.path.abspath('../../'))
from backend.models import Dataset, Document, Summary, SummaryGroup, AnnotationResult, DocStatus
from backend.app import create_app

DELETE_UNVALID_DATA=False

#%%
# Loading data from database
app = create_app()
db = SQLAlchemy(app)
#q_results = db.session.query(Document, Dataset) \
#    .join(Dataset) \
#    .filter(Dataset.name == 'BBC').all()

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
q_results = db.session.query(Document, Dataset) \
    .join(Dataset).all()

df_doc_prop = pd.DataFrame([])
summaries = {}; rec_id_2_doc_id = {}
for doc, _ in q_results:
    doc_json = json.loads(doc.doc_json)
    df_doc_prop = df_doc_prop.append(parse(doc_json)\
        .assign(doc_id=doc_json['doc_id'])\
        .assign(rec_id=doc.id))
    summaries[doc.id] = doc.summary
    rec_id_2_doc_id[doc.id] = doc_json['doc_id']


# Contains data of the document with the summary
df_doc = pd.DataFrame(df_doc_prop[df_doc_prop['type'] != 'whitespace'].groupby('rec_id').count())
df_doc = df_doc.rename(columns={'index': 'word_len'}).drop(columns=['type', 'content'])
df_doc['summ'] = df_doc.index.map(summaries.get)

#%%
# Retrieve highlights
def process_doc(results, word_idx, doc_id):
    """
    Build indexes and texts for the given document
    """
    indexes = []
    texts = []
    result_ids = []
    doc_id = doc_json['doc_id']
    for result_id, data in results.items():
        if result_id not in valid_res:
            print (result_id + " IS NOT QUALIFIED")
            continue
    for result_id, data in results:
        for h_id, highlight in data.items():
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
for doc, _ in q_results:
    doc_json = json.loads(doc.doc_json)
    if 'results' not in doc_json:
        continue
    word_idx = df_doc_prop.groupby('rec_id').get_group(doc.id)
    pdoc = process_doc(doc_json, word_idx)
    if pdoc is not None:
        df_h = df_h.append(pdoc.assign(rec_id=doc.id))


from itertools import chain

df_doc = df_doc.assign(doc_text=lambda x: df_doc_prop[df_doc_prop['type'] == 'word'].groupby('rec_id')['content'].apply(list))
df_doc = df_doc.assign(h_idxs=lambda x: df_h.groupby('rec_id').apply(lambda x: list(set(chain(*x.indexes)))))

for i, h_idx in enumerate(df_doc['h_idxs']):
    if not isinstance(h_idx, list):
        df_doc['h_idxs'][i+1] = []

df_doc = df_doc.assign(doc_idxs=lambda x: df_doc_prop[df_doc_prop['type']=='word'].groupby('rec_id')['index'].apply(list))

df_doc = df_doc.assign(h_len=lambda x: df_doc[['h_idxs']].apply(lambda x: len(x['h_idxs']), axis=1))

df_doc = df_doc.assign(
    doc_h_overlap=lambda x: df_doc[['h_idxs', 'doc_idxs']]
        .apply(lambda x: set(x['doc_idxs']) & set(x['h_idxs']), axis=1)
        .apply(lambda x: len(list(x))))

df_doc = df_doc.assign(
    doc_h_overlap_recall=lambda x: df_doc.doc_h_overlap/df_doc.word_len)

df_doc = df_doc.assign(
    doc_h_overlap_precision=lambda x: df_doc.doc_h_overlap/df_doc.h_len)

df_doc = df_doc.assign(
    doc_h_overlap_F1=lambda x: 2*df_doc.doc_h_overlap_recall*df_doc.doc_h_overlap_precision/(df_doc.doc_h_overlap_recall+df_doc.doc_h_overlap_precision))


#%%
# Calculate word overlap ratio between summary and highlights

df_doc = df_doc.assign(h_text=lambda x: df_h.groupby('rec_id').apply(lambda x: ' '.join(x.text).split()))

for i, h_t in enumerate(df_doc['h_text']):
    if not isinstance(h_t, list):
        df_doc['h_text'][i+1] = []

print (df_doc['h_text'])


from nltk.util import ngrams
for i in range(1, 4):
    kwargs_1 = {
        'summ_h_%s_gram' % i: lambda x: df_doc[['summ', 'h_text']].apply(
        lambda x: set(ngrams(x['h_text'], i)) & set(ngrams(x['summ'], i)), axis=1).apply(lambda x: len(list(x)))
    }
    kwargs_2 = {
        'summ_h_overlap_%s_gram_recall' % i: lambda x: df_doc[['h_text', 'summ_h_%s_gram' % i]].apply(lambda x: x['summ_h_%s_gram' % i] / (len(list(set(ngrams(x['h_text'], i))))+1), axis=1)
    }
    kwargs_3 = {
        'summ_h_overlap_%s_gram_precision' % i: lambda x: df_doc[['summ', 'summ_h_%s_gram' % i]].apply(
            lambda x: x['summ_h_%s_gram' % i] / len(list(set(ngrams(x['summ'], i)))), axis=1)
    }
    kwargs_4 = {
        'summ_h_overlap_%s_gram_F1' % i: lambda x: 2*df_doc['summ_h_overlap_%s_gram_precision' % i]*df_doc['summ_h_overlap_%s_gram_recall' % i]/(df_doc['summ_h_overlap_%s_gram_precision' % i]+df_doc['summ_h_overlap_%s_gram_recall' % i])
    }
    df_doc = df_doc.assign(**kwargs_1)
    df_doc = df_doc.assign(**kwargs_2)
    df_doc = df_doc.assign(**kwargs_3)
    df_doc = df_doc.fillna(0)
    df_doc = df_doc.assign(**kwargs_4)
    df_doc = df_doc.fillna(0)

#%%
# Calculate word overlap ratio between summary and doc
for i in range(1, 4):
    kwargs_1 = {
        'summ_doc_%s_gram' % i: lambda x: df_doc[['summ', 'doc_text']].apply(
        lambda x: set(ngrams(x['doc_text'], i)) & set(ngrams(x['summ'], i)), axis=1).apply(lambda x: len(list(x)))
    }
    kwargs_2 = {
        'summ_doc_overlap_%s_gram_recall' % i: lambda x: df_doc[['doc_text', 'summ_doc_%s_gram' % i]].apply(lambda x: x['summ_doc_%s_gram' % i] / len(list(set(ngrams(x['doc_text'], i)))), axis=1)
    }
    kwargs_3 = {
        'summ_doc_overlap_%s_gram_precision' % i: lambda x: df_doc[['summ', 'summ_doc_%s_gram' % i]].apply(
            lambda x: x['summ_doc_%s_gram' % i] / len(list(set(ngrams(x['summ'], i)))), axis=1)
    }
    kwargs_4 = {
        'summ_doc_overlap_%s_gram_F1' % i: lambda x: 2*df_doc['summ_doc_overlap_%s_gram_precision' % i]*df_doc['summ_doc_overlap_%s_gram_recall' % i]/(df_doc['summ_doc_overlap_%s_gram_precision' % i]+df_doc['summ_doc_overlap_%s_gram_recall' % i])
    }
    df_doc = df_doc.assign(**kwargs_1)
    df_doc = df_doc.assign(**kwargs_2)
    df_doc = df_doc.assign(**kwargs_3)
    df_doc = df_doc.fillna(0)
    df_doc = df_doc.assign(**kwargs_4)
    df_doc = df_doc.fillna(0)

#%%
# Saving result to csv files
import numpy as np
import os

result_path = './'

df_doc.to_csv(os.path.join(result_path, 'df_doc.csv'))

df_result = pd.DataFrame(df_doc.describe(include=[np.number]))
df_result.to_csv(os.path.join(result_path, 'df_result.csv'))

#%%
# Retrieve result per coder and store them in DataFrame
q_results = db.session.query(Document).all()

df_annotations = pd.DataFrame([])
df_doc_prop_group = df_doc_prop.groupby('rec_id')
count = 0
test = {}
for doc in q_results:
    res_json = json.loads(doc.doc_json)
    if 'results' not in res_json:
        continue
    results = res_json['results']
    for result_id, result in results.items():
        if result_id not in valid_res:
            print (result_id + " IS NOT QUALIFIED")
            continue
        highlights = result['highlights']
        indexes = []
        texts = []
        word_idx = df_doc_prop_group.get_group(doc.id)
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
            }).assign(doc_id=doc.doc_id, result_id=result_id, rec_id=doc.id))
#%%
# Create the Fleiss' kappa matrix
df_agreement = pd.DataFrame([])

result_ids = {}
kappa_docs = {}
for doc_id, data in df_annotations.groupby('rec_id'):
    print('Processing %s' % doc_id)
    doc_idxs = list(df_doc.loc[doc_id]['doc_idxs'])
    df_result = data.groupby('result_id')
    x_idxs = [list(chain(*result['indexes']))
              for _, result in df_result]
    print (len(x_idxs))
    x_texts = [list(chain(result['texts'])) for _, result in df_result]
    # Assigning index to each unique result_id
    count_i = 0
    store_i = -1
    for s_i in list(data['result_id']):
        if s_i in valid_res:
            store_i = s_i
    if store_i == -1:
        print ("NOTHING IS VALIDED, SKIP THIS ARTICLE")
        continue
    result_id2idx = {}
    result_idx2id = {}
    for i in list(data['result_id']):
        if i != store_i and (i not in valid_res):
            count_i += 1
            store_i = i
        result_id2idx[store_i] = count_i
        result_idx2id[count_i] = store_i

    result_ids[doc_id] = data.assign(idx=lambda x: data['result_id'].apply(lambda y: result_id2idx.get(y)))
    # Start building Fleiss' Kappa
    coder_m = pd.DataFrame([[1 if idx in c else 0 for idx in doc_idxs] for c in x_idxs]).T
    coder_m.index = doc_idxs
    # Calculate P_e of Fleiss' Kappa
    cat_m = pd.DataFrame([])
    cat_m['c_1'] = coder_m.sum(axis=1)
    cat_m['c_0'] = (len(x_idxs) - coder_m.sum(axis=1))
    cat_m_sum = cat_m.sum(axis=0) / (len(x_idxs) * len(doc_idxs))
    import math

    P_e = math.pow(cat_m_sum['c_1'], 2) + math.pow(cat_m_sum['c_0'], 2)

    # Commented this section because Kappa for each document is the same with mean of the following approach
    # Calc Fleiss' Kappa for each document
    # import numpy as np
    #
    # pi_m = pd.DataFrame(np.zeros((len(x_idxs), len(x_idxs))))
    # P_i = []
    # for i in range(len(doc_idxs)):
    #     sum_cat_ij = 0
    #     for j in range(2):
    #         sum_cat_ij += \
    #             cat_m.iloc[i][j] * (cat_m.iloc[i][j] - 1)
    #     P_i.append(1 / (len(x_idxs) * (len(x_idxs) - 1)) * sum_cat_ij)
    # P_o = sum(P_i) / len(doc_idxs)
    # kappa_docs[doc_id] = (P_o - P_e) / (1 - P_e)

    # Calc Fleiss' Kappa for each pair of possible combination
    from itertools import product
    import numpy as np

    cartesian = product(range(len(x_idxs)), repeat=2)
    pi_m = pd.DataFrame(np.zeros((len(x_idxs), len(x_idxs))))
    for idx in cartesian:
        if idx[0] == idx[1]:
            continue
        new_x_idxs = []
        new_x_idxs.append(x_idxs[idx[0]])
        new_x_idxs.append(x_idxs[idx[1]])
        coder_m = pd.DataFrame([[1 if idx in c else 0 for idx in doc_idxs] for c in new_x_idxs]).T
        coder_m.index = doc_idxs
        cat_m = pd.DataFrame([])
        cat_m['c_1'] = coder_m.sum(axis=1)
        cat_m['c_0'] = (len(new_x_idxs) - coder_m.sum(axis=1))
        P_i = []
        for i in range(len(doc_idxs)):
            sum_cat_ij = 0
            for j in range(2):
                sum_cat_ij += \
                    cat_m.iloc[i][j] * (cat_m.iloc[i][j] - 1)
            P_i.append(
                1 / (len(new_x_idxs) * (len(new_x_idxs) - 1)) * sum_cat_ij)
        P_o = sum(P_i) / len(doc_idxs)
        kappa = (P_o - P_e) / (1 - P_e)
        pi_m[idx[0]][idx[1]] = kappa
    pi_m.sum(axis=1) / (len(x_idxs) - 1)
    kwargs = {
        str(doc_id): pi_m.sum(axis=1) / (len(x_idxs) - 1),
        '%s_len' % doc_id: pd.Series([len(x) for x in x_idxs]),
        '%s_text' % doc_id: pd.Series([x for x in x_texts]),
        '%s_r_id' % doc_id: pd.Series([result_idx2id.get(i) for i in range(len(x_idxs))])
    }
    df_agreement = df_agreement.assign(**kwargs)

#%%
# Save agreement to files
df_agreement_describe = df_agreement.describe()
df_agreement_describe = df_agreement_describe.append(df_doc['doc_idxs'].apply(lambda x: len(x)))
df_agreement.to_csv(os.path.join(result_path, 'df_agreement.csv'))
df_annotations.to_csv(os.path.join(result_path, 'df_annotations.csv'))
df_agreement_describe.to_csv(os.path.join(result_path, 'df_agreement_desc.csv'))
for key, value in result_ids.items():
    save_path = os.path.join(result_path, 'result_ids')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    value.to_csv(os.path.join(
            save_path, '%s_result_ids.csv' % key))



