from models.tf_model import FrequencyModel
from ir_measures import ScoredDoc
import numpy as np
from utils import Query


class OkapiBM25Model(FrequencyModel):
    def __init__(self):
        super().__init__()
        self.k1 = 1.2
        self.b = 0.75
        self.k3 = 7

    def wt(self, query_tf : dict[int, float] , term: int, doc: int):
        nom = (
            self.idf[term]
            * (self.k1 + 1)
            * self.tf[term][doc]
            * (self.k3 + 1)
            * query_tf[term]
        )
        den = (
            self.k1 * ((1 - self.b ) + self.b * (self.doc_len[doc] / self.avg_doc_len))
            + self.tf[term][doc]
        ) * (self.k3 + query_tf[term])
        return nom / den
    
    def wt_f(self, query_tf : dict[int, float], term: int, doc: int, VR: list[str] , Vdoc : list[str]):
        V = set([d.doc_id for d in Vdoc])
        VNR = list(V - set(VR))

        VRt = 0
        for d in VR:
            if int(d) in self.tf[term]:
                VRt += 1
        VNRt = 0
        for d in VNR:
            if int(d) in self.tf[term]:
                VNRt += 1

        nom = (VRt + 1/2) / (VNRt + 1/2)
        den = (len(self.tf[term]) - VRt + 1/2) / ( self.corpus_size - len(self.tf[term]) - len(VR) + VRt + 1/2)
        return np.log10((nom/den) * (self.wt(query_tf, term, doc) / self.idf[term]))

    def tf_function(self, term: int, doc: int):
        return self._tf_n(term, doc)

    def idf_function(self, term: int):
        return self._idf_p(term, 0.5)

    def retrieve_query(self, query: Query):
        return [
            ScoredDoc(query.id, str(doc), score)
            for score, doc in self.daat_kretrieve_okapi(query, 15, 0)
        ]

    def retrieve_feedback(self, query: Query, rdocs: list[str], alldocs: list[str]):
        return [
            ScoredDoc(query.id, str(doc), score)
            for score, doc in self.daat_kretrieve_okapi(query, 15, 1, rdocs, alldocs)
        ]    
