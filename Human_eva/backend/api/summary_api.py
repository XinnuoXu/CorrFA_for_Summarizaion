import http
from flask import jsonify, request

from . import api
from backend.models import SummariesPair, SummaryGroup, Summary


@api.route('/summary/<doc_id>', methods=['GET'])
def api_summary_get(doc_id):
    if request.method == 'GET':
        return Summary.get_text(doc_id)

@api.route('/summ_group', methods=['GET'])
def api_summ_group_get_names():
    summ_groups = SummaryGroup.query.all()
    if not summ_groups:
        return '', http.HTTPStatus.NO_CONTENT
    else:
        result = dict()
        result['names'] = []
        for summ_group in summ_groups:
            result['names'].append(summ_group.name)
        return jsonify(result)
