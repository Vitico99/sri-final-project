from tf_model import FrequencyModel
from ir_measures import ScoredDoc


class OkapiBM25Model(FrequencyModel):
    def __init__(self):
        super().__init__()
        self.k1 = 1.2
        self.b = 0.75
        self.k3 = 7

    def score(self, term, doc):
        nom = (
            self.idf[term]
            * (self.k1 + 1)
            * self.tf[term][doc]
            * (self.k3 + 1)
            * self.tf[term][doc]
        )
        den = (
            self.k1 * (1 - self.b + self.b * (self.doc_len[doc] / self.avg_doc_len))
            + self.tf[term][doc]
            + self.tf[term][doc] * self.k3
        ) * (self.k3 + self.tf[term][doc])
        return nom / den

    def tf_function(self, term, doc):
        return self._tf_n(term, doc)

    def idf_function(self, term):
        return self._idf_p(term, 0.5)

    def retrieve(self, query):
        return [
            ScoredDoc(query.id, str(doc), score)
            for score, doc in self.daat_kretrieve(query, 10)
        ]
