import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import joblib
#from sklearn.ensemble import RandomForestClassifier


def spam_results(content = None, num_features = 10):
    """
    content: string
    Article to be passed in the model
    default: None
    
    num_features: int
    total words to consider for weights
    default: 10
    """
    
    pipeline = joblib.load('tfidf_rfc.pkl')
    
    class_names=['0','1']
    
    """
    class 0 -> NOT SPAM
    class 1 -> SPAM
    """
    
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(content, pipeline.predict_proba, num_features = num_features)
    
    probability = pipeline.predict_proba([content])
    
    results = {}
    
    results['weights'] = exp.as_list()
    
    results['probability'] = probability
    
    if probability[0][0] > probability[0][1]:
        results['predicted_class'] = '0'
    else:
        results['predicted_class'] = '1'
    return results


