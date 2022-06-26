import collections
from regex import W
from models.tf_model import FrequencyModel
from ir_measures import ScoredDoc
import numpy as np
from collections import Counter
from utils import Query


class ClassVectorSpaceModel(FrequencyModel):
    """ Implementation of the vector space model studied in class """
    def fit(self, corpus):
        super().fit(corpus)
        self.doc_norm = { doc : 0 for doc in self.docs }
        for term in self.tf:
            for doc in self.tf[term]:
                self.doc_norm[doc] += (self.tf[term][doc] * self.idf[term]) ** 2
        self.doc_norm = { doc : np.sqrt(norm) for doc, norm in self.doc_norm.items() }
                
    def dscore(self, term : int, doc : int):
        try:
            return self.tf[term][doc] * self.idf[term]
        except:
            return 0
       
    def wdoc(self, doc: int):
        w = {}
        for term in self.tf:
            w[term] = self.dscore(term, doc)
        return w 

    def wquery(self, query_tf: dict[int, int]):
        max_tf = max(query_tf.values())
        w_tq = {}
        norm = 0
        for term in query_tf:
            w = ( 0.4 + (1 - 0.4 ) * (query_tf[term] / max_tf)) * self.idf[term]
            w_tq[term] = w
            norm += w ** 2 
        norm = np.sqrt(norm)
        return { term : w_tq[term] / norm for term in query_tf }
    
    def tf_function(self, term : int, doc : int):
        return self._tf_m(term, doc)

    def idf_function(self, term: int):
        return self._idf_t(term, 1)

    def retrieve_query(self, query: Query):
        query_id = query.id
        # replacing query words for terms to not repeat this step in the loop
        query = [self.vocabulary[word] for word in query if word in self.vocabulary]
        if len(query) == 0:
            return []
        query_tf = Counter(query) 
        wquery = self.wquery(query_tf)
        return [
            ScoredDoc(query_id, str(doc), score)
            for score, doc in self.daat_kretrieve(wquery, 15)
        ]

    def retrieve_feedback(self, query: Query, rdocs: list[str] , alldocs : list[str]):
        wquery = self.feedback(query, rdocs, alldocs)
        
        return [
            ScoredDoc(query.id, str(doc), score)
            for score, doc in self.daat_kretrieve(wquery, 15)
        ]

    def feedback(self, query: Query, rdocs : list[str] , alldocs : list[str]):
        query_int = [self.vocabulary[word] for word in query if word in self.vocabulary]
        query_tf = Counter(query_int) 
        wquery = self.wquery(query_tf)

        iddocs = set( [d.doc_id for d in alldocs])
        nrdocs = list(iddocs - set(rdocs))

        wrdocs = []
        wnrdocs = []
        for doc in rdocs:
            wrdocs.append(self.wdoc(int(doc)))
        for doc in nrdocs:
            wnrdocs.append(self.wdoc(int(doc)))
        
        new_query = {}

        beta = 0.75
        gamma = 0.25
        alpha = 0.75
        sumR = {}
        sumNR = {}
        for term in self.tf:
            sumR[term] = 0
            sumNR[term] = 0
            for i in range(len(rdocs)):
                sumR[term] += wrdocs[i][term] 
            for i in range(len(nrdocs)):
                sumNR[term] += wnrdocs[i][term] 

            try:
                wq = wquery[term]
            except:
                wq = 0

            r = alpha * wq + sumR[term] * beta / len(rdocs) -  sumNR[term] * gamma / len(nrdocs)
            new_query[term] = 0
            if r > 0:
                new_query[term] = r
        
        return new_query
