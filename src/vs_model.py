from regex import W
from tf_model import FrequencyModel
from ir_measures import ScoredDoc
import numpy as np


class ClassVectorSpaceModel(FrequencyModel):
    """Implementation of the vector space model studied in class"""
    def fit(self, corpus) -> None:
        super().fit(corpus)
        self.doc_norm = { doc : 0 for doc in self.docs }
        for term in self.tf:
            for doc in self.tf[term]:
                self.doc_norm[doc] += (self.tf[term][doc] * self.idf[term]) ** 2
        self.doc_norm = { doc : np.sqrt(norm) for doc, norm in self.doc_norm.items() }
                

    def dscore(self, term, doc):
        return self.tf[term][doc] * self.idf[term] / self.doc_norm[doc]
          
    def qscore(self, query_tf : dict[int, int]):
        max_tf = max(query_tf.values())
        w_tq = {}
        norm = 0
        for term in query_tf:
            w = 0.5 + 0.5 * query_tf[term] / max_tf * self.idf[term]
            w_tq[term] = w
            norm += w ** 2 
        norm = np.sqrt(norm)
        return lambda term: w_tq[term] / norm

    def tf_function(self, term, doc):
        return self._tf_m(term, doc)

    def idf_function(self, term):
        return self._idf_t(term, 1)

    def retrieve(self, query):
        return [
            ScoredDoc(query.id, str(doc), score)
            for score, doc in self.daat_tretrieve(query, 0.1)
        ]
