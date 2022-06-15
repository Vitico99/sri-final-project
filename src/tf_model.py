import numpy as np
from collections import Counter
from utils import Doc


class FrequencyModel:
    def __init__(self):

        # Maps word to an unique int for more efficient computations this unique int
        # will be the representation of a term
        self._vocabulary = {}

        # An inverted index of term frequency with the structure
        # { term : { doc : freq } }
        self._tf = {}

        # Associates each term with its idf value
        self._idf = {}

        # Associates each term with its document frequency, this field is used
        # just for explainability in the formulas
        self._df = {}

        # Associates each document with the total number of terms in the doc,
        # this measure is used for normalization purposes
        self._doc_len = {}

        # Associantes each document with the largest number of ocurrences of
        # a term in the document, this measure is used for normalization purposes
        self._doc_max_tf = {}

        # Total number of documents in the corpus
        self._corpus_size = 0

        # Average number of terms in a document in the corpus
        self._avg_doc_len = 0

        # contains the ids of the documents present in the corpus
        self._docs = []

    def fit(self, corpus, normalized_tf=False):
        for doc in corpus:

            self._corpus_size += 1
            self._docs.append(doc.id)

            self._doc_len[doc.id] = len(doc.words)
            self._avg_doc_len += len(doc.words)
            self._doc_max_tf[doc.id] = 0

            # Construct a dictionary { word : frequency } for the document
            doc_tf = Counter(doc.words)

            for word, freq in doc_tf.items():

                if word not in self._vocabulary:
                    # Register the word as a term
                    term = self._vocabulary[word] = len(self._vocabulary)
                    self._tf[term] = {}
                else:
                    term = self._vocabulary[word]

                self._tf[term][doc.id] = freq
                self._doc_max_tf[doc.id] = max(self._doc_max_tf[doc.id], freq)

        self._avg_doc_len /= self._corpus_size

        for term in self._tf:
            self._df[term] = len(self._tf[term])
            self._idf[term] = self.idf(term)

            if normalized_tf:
                self._tf[term] = {
                    doc: self.tf_norm(term, doc) for doc in self._tf[term]
                }

    def tf_norm(self, term, doc):
        return self._tf_smooth_norm(term, doc, 0.4)

    def _tf_log_norm(self, term, doc):
        return np.log10(1 + self._tf[term][doc])

    def _tf_maxtf_norm(self, term, doc):
        return self._tf[term][doc] / self._doc_max_tf[doc]

    def _tf_smooth_norm(self, term, doc, a):
        return a + (1 - a) * (self._tf[term][doc] / self._doc_max_tf[doc])

    # ========================== Multiple definitions of idf ====================================

    def idf(self, term):
        return self._idf_base(term)

    def _idf_base(self, term):
        return np.log10(self._corpus_size / self._df[term])

    def _idf_smooth(self, term):
        return np.log10(self._corpus_size / (1 + self._df[term])) + 1

    def _pidf_smooth(self, term, a):
        return np.log10((self._corpus_size - self._df[term] + a) / (self._df[term] + a))

    # ==============================================================================================

    def score(self, term, doc):
        raise NotImplementedError()

    def retrieve(self, query):
        raise NotImplementedError()
