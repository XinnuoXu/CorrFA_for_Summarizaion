import json
import os
import sys

sys.path.append(os.path.abspath('../'))
from backend.models import Document, Dataset, Summary, SummaryGroup, SummariesPair, User
from backend.app import create_app
from flask_sqlalchemy import SQLAlchemy

dataset_path = '../backend/BBC_pair'
def example_filter(sentence):
    toks = sentence.split(" ")
    if toks.count("<strong>") < 2:
        return True
    return False

def one_split(db, idx, sanity_data):
    # Insert dataset
    dataset = Dataset(name="ALG_FACT"+str(idx))
    db.session.add(dataset)
    db.session.commit()

    summaries_path = os.path.join(dataset_path, 'summaries')
    documents_path = os.path.join(dataset_path, 'documents')
    for doc_id in sanity_data:
        file_name = doc_id + ".data"
        file_path = os.path.join(documents_path, file_name)
        summ_path = os.path.join(summaries_path, file_name)
        with open(summ_path, 'r') as infile:
            summ_json = json.load(infile)
        with open(file_path, 'r') as infile:
            json_result = json.load(infile)
            did = json_result['doc_id']
            for i, item in enumerate(summ_json):
                if item['name'].find("|||") == -1:
                    continue
                if example_filter(item['text']):
                    continue
                document = Document(
                    dataset_id=dataset.id,
                    doc_id=json_result['doc_id'],
                    doc_json=json.dumps(json_result),
                    summary=json.dumps(item),
                    sanity_statement=sanity_data[did]["sanity_statement"],
                    sanity_answer=sanity_data[did]["sanity_answer"]
                )
                db.session.add(document)
                db.session.commit()


def init_database_split(db, num_of_split):
    dataset_name = os.path.split(dataset_path)[1]
    sanity_path = os.path.join(dataset_path, 'sanity_id/sanity.txt')

    sanity_data = []
    for i in range(num_of_split):
        sanity_data.append({})

    for i, line in enumerate(open(sanity_path)):
        flist = line.strip().split("\t")
        split_id = i % num_of_split
        sanity_data[split_id][flist[0]] = {"sanity_answer": bool(int(flist[2])), "sanity_statement": flist[1]}

    # Insert documents
    for i in range(num_of_split):
        one_split(db, i, sanity_data[i])

def init_database(db):
    # user = User(email='admin@localhost', password='localhost')
    # db.session.add(user)
    # db.session.commit()
    dataset_path = '../backend/BBC_pair'
    dataset_name = os.path.split(dataset_path)[1]

    summaries_path = os.path.join(dataset_path, 'summaries')
    documents_path = os.path.join(dataset_path, 'documents')
    sanity_path = os.path.join(dataset_path, 'sanity_id/sanity.txt')

    # Existing dataset
    #dataset = db.session.query(Dataset).filter_by(name='BBC').first()
    # Insert dataset
    dataset = Dataset(name="BBC")
    db.session.add(dataset)
    db.session.commit()

    sanity_data = {}
    for line in open(sanity_path):
        flist = line.strip().split("\t")
        sanity_data[flist[0]] = {"sanity_answer": bool(int(flist[2])), "sanity_statement": flist[1]}

    # Insert documents
    for file in os.listdir(documents_path):
        file_path = os.path.join(documents_path, file)
        summ_path = os.path.join(summaries_path, file)
        with open(summ_path, 'r') as infile:
            summ_json = json.load(infile)
        with open(file_path, 'r') as infile:
            json_result = json.load(infile)
            did = json_result['doc_id']
            for i, item in enumerate(summ_json):
                document = Document(
                    dataset_id=dataset.id,
                    doc_id=json_result['doc_id'],
                    doc_json=json.dumps(json_result),
                    summary=json.dumps(item),
                    sanity_statement=sanity_data[did]["sanity_statement"],
                    sanity_answer=sanity_data[did]["sanity_answer"]
                )
                db.session.add(document)
                db.session.commit()

if __name__ == '__main__':
    app = create_app()
    db_app = SQLAlchemy(app)
    #init_database(db_app)
    init_database_split(db_app, 10)
