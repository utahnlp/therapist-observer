# Time-stamp: <2019-06-04 15:16:52 jcao>
# --------------------------------------------------------------------
# File Name          : feature_extractor.py
# Original Author    : jiessie.cao@gmail.com
# Description        : A feature extractor to get useful features from dataset
# --------------------------------------------------------------------


import ujson as json, sys, os, numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

class feature_extractor(object):
    def __init__(self):
        #lda number of topics
        self.n_components = 10

        #kmeans number of clusters
        self.n_clusters = 100

        self.lda_pipe = None
        self.kmeans_pipe = None

    def train(self, input_json):
        dirname = os.path.dirname(input_json)
        model_path = os.path.join(dirname, 'model.pkl')

        print "Loading data for cluster training..."
        docs = self._load_docs(input_json)


        tfidf_vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 2, stop_words = 'english')
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_jobs = 8)
        kmeans_pipe = Pipeline([('tfidf', tfidf_vectorizer), ('kmeans', kmeans)])

            
        """
        count_vectorizer = CountVectorizer(max_df = 0.5, min_df = 2, stop_words = 'english')
        lda = LatentDirichletAllocation(n_components = self.n_components, max_iter = 1, learning_method = 'batch', n_jobs = 10)
        lda_pipe = Pipeline([('count', count_vectorizer), ('lda', lda)] )
        """


        print("Training kmeans clustering algorithm...")
        kmeans_pipe.fit(docs)

        """
        print("Training LDA...")
        lda_pipe.fit(docs)
        """

        self.kmeans_pipe = kmeans_pipe
        #self.lda_pipe = lda_pipe

        print "dumping model to: %s" % model_path
        #joblib.dump((self.kmeans_pipe, self.lda_pipe), model_path)
        joblib.dump(self.kmeans_pipe, model_path)


    #return cluster id and lda topics probability contatenated as a single array
    def extract_features(self,docs):
        km = self.kmeans_pipe.predict(docs)
        #ld = self.lda_pipe.transform(docs)

        return km.reshape(-1, 1)
        #return np.concatenate((km.reshape(-1, 1), ld), axis = 1)


    #Load existing model from path
    def load_model(self, model_path):
        print "Loading model: %s" %model_path
        #self.kmeans_pipe, self.lda_pipe = joblib.load(model_path)
        self.kmeans_pipe = joblib.load(model_path)


    #load documents for training, usd by train()
    def _load_docs(self, input_json):
        docs = []
        for l in tqdm(open(input_json)):
            sample = json.loads(l)
            for option in sample['options-for-correct-answers']:
                doc = ' '.join(option['tokenized_utterance'])
                docs.append(doc)
            for option in sample['options-for-next']:
                doc = ' '.join(option['tokenized_utterance'])
                docs.append(doc)
            for msg in sample['messages-so-far']:
                doc = ' '.join(option['tokenized_utterance'])
                docs.append(doc)
        return list(set(docs))
