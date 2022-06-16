from tf_model import FrequencyModel
from ir_measures import ScoredDoc


class ClassVectorSpaceModel(FrequencyModel):
    """Implementation of the vector space model studied in class"""

    def score(self, term, doc):
        return self.tf[term][doc] * self.idf[term] / self.doc_len[doc]

    def tf_function(self, term, doc):
        return self._tf_m(term, doc)

    def idf_function(self, term):
        return self._idf_t(term)

    def retrieve(self, query):
        return [
            ScoredDoc(query.id, str(doc), score)
            for score, doc in self.daat_kretrieve(query, 10)
        ]
