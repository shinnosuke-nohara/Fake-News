import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import joblib
#from sklearn.ensemble import RandomForestClassifier
import json
import requests
from tqdm import tqdm
import time
from datetime import datetime
from elasticsearch import Elasticsearch
from random import choice


def spam_results(content=None, num_features=10):
    """
    content: string
    Article to be passed in the model
    default: None

    num_features: int
    total words to consider for weights
    default: 10
    """

    pipeline = joblib.load('tfidf_rfc.pkl')

    class_names = ['0', '1']

    """
    class 0 -> NOT SPAM
    class 1 -> SPAM
    """

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        content, pipeline.predict_proba, num_features=num_features)

    probability = pipeline.predict_proba([content])

    weights = exp.as_list()
    weights_list = []

    for weight in weights:
        weights_list.append({"word": weight[0], "score": weight[1]})

    results = {}

    results['weights'] = weights_list

    results['probability'] = [probability[0][0], probability[0][1]]

    if probability[0][0] > probability[0][1]:
        results['predicted_class'] = '0'
    else:
        results['predicted_class'] = '1'
    return results


def get_data_from_db(db):
    host = 'http://15.207.24.247:9200/' + db + '/_search'
    json_body = '''{
        "query": {
                "bool": {
                    "must_not": {
                        "exists":{
                            "field":"fake_news"
                            }
                        }
                    }
                }
    }'''
    headers = {
        'Content-Type': 'application/json',
    }
    params = {
        'size': 100
    }
    resp = requests.get(host, params=params, headers=headers, data=json_body)
    resp_text = json.loads(resp.text)
    document_list = []

    for data in resp_text['hits']['hits']:
        try:
            content_list = {}
            content_list['id'] = data['_id']
            content_list['content'] = data['_source']['Content']
            document_list.append(content_list)
        except Exception as err:
            print(err)

    return document_list


def writeInDB(elastic_client, doc_id, doc, db):
    source_to_update = {"doc": {"fake_news": doc}}
    response = elastic_client.update(
        index=db,
        doc_type="_doc",
        id=doc_id,
        body=source_to_update
    )
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), response['result'])
    if response['result'] == "updated":
        pass
        # print ("result:", response['result'])
        # print ("Update was a success for ID:", response['_id'])
        # print ("New data:", source_to_update)
    else:
        print("result:", response['result'])
        print("Response failed:", response['_shards']['failed'])

    return response


def write_data_in_db(id, result, db):
    host = 'http://15.207.24.247:9200/'+db+'/_update/'+str(id)
    headers = {
        'Content-Type': 'application/json',
    }
    post_body = {
        "doc": {
            "fake_news": result
        },
        "detect_noop": False
    }

    post_body = json.dumps(post_body)
    response = requests.post(url=host, headers=headers, data=post_body)
    return response


while True:
    elastic_client = Elasticsearch([{'host': '15.207.24.247', 'port': 9200}])
    db = choice(['news_list', 'news_data'])
    data = get_data_from_db(db)
    # print(data)

    cnt = 0
    start = time.time()
    for ele in tqdm(data):
        text = ele['content']
        ans = spam_results(text)

        print('->', ans)

        res = writeInDB(elastic_client, ele['id'], ans, db)
        # res = write_data_in_db(ele['id'], ans, db)

       # print(res)
        cnt = cnt+1
       # break

    print("*"*40)
    print(time.time()-start)
    print(cnt)
    print("*"*40)
