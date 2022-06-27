# Developer Guide

## 1. Setting up the developing environment

Clone this repository and run the following command in the terminal:

```bash
make build
```

Then set up the `pre-commit` framework for `black` [code formatter](https://github.com/psf/black).

```
pre-commit install
```

Run:

```
streamlit run main.py
```

## 2. Architecture Overview

An Information Retrieval System is defined as a tuple $(D, Q, F, R)$ where:

- $D$ are the logic representations of documents.
- $Q$ are the logic representations of user needs, defined as queries.
- $F$ is a framework to define the relationship between documents and queries.
- $R$ is a ranking function $r : D \times Q \to \mathbb{R}$.

This project follows this structure allowing to replace any of the components in the IRS without affecting the rest.


```python
# src/irs.py

class IRSystem:
    def __init__(self, model: IRModel, text_processor: TextProcessor) -> None:
        self.model = model
        self.text_processor = text_processor
        self.model.fit(self._docs_iter())

    def docs_iter(self) -> Iterable[Doc]:
        raise NotImplementedError()

    def retrieve(self, query) -> list[ScoredDoc]:
        raise NotImplementedError()

    def register_query(self, text) -> int:
        raise NotImplementedError()

    def eval(self, measures: list[Measure]) -> dict[Measure, float]:
        raise NotImplementedError()
```

### 2.1. Document representation

Lets begin with `_docs_iter` in the `IRSystem` class, this method is the representation of $D$ and provides an iterator over the corpus for efficient memory use. In this method you would process the files using NLP techniques to tokenize the document then representing it with the class `Doc`.

```python
# src/utils.py

class Doc:
    def __init__(self, id, words) -> None:
        self.id = id
        self.words = words
    
    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)
```

### 2.2. Query representation

Query representation is the same as doc:

```python
# src/utils.py

class Query:
    def __init__(self, id, words) -> None:
        self.id = id
        self.words = words
    
    def __iter(self):
        return iter(self.words)
```
In `register_query` we provide a very basic implementation of how to register a query in the system, just process the text and generate 
an unique identifier for the query.

```python
    def register_query(self, text) -> int:
        words = self.text_processor.process(query)
        return Query(uuid1(), words)
```

### 2.3 Framework and Ranking

To discuss the framework we must first talk about IR models, the models combine elements $F$ and $R$ of the IR system.

```python
#src/tf_model.py

class IRModel:
    def fit(self, corpus: iter[Doc]):
        raise NotImplementedError()

    def retrieve_query(self, query: Query):
        raise NotImplementedError()
    
    def retrieve_feedback(self, query_id: int, wquery: 			dict[int, float]): 
         raise NotImplementedError()
```
`fit` method is the responsable of constructing the needed data structures for the framework $F$, in this case we chose the weighted scheme $tf-idf$ framework.

The `retrieve_query` method is in charge of retrieving the initial set of relevant documents.
Once the user marks the relevant ones from the retrieved set, the set is refined using the `retrieve_feedback` method.

```python
# src/tf_model.py

class FrequencyModel(IRModel):
    def __init__(self, normalize:bool=False):

        # Data structures for frequency models
        self.vocabulary: dict[str, int] = {}
        self.docs: list[int] = []
        self.tf : dict[int, dict[int, float]] = {}
        self.idf: dict[int, float] = {}
        self.df: dict[int, int] = {}
        self.doc_len: dict[int, int] = {}
        self.doc_max_tf: dict[int, int] = {}
        self.doc_avg_tf: dict[int, float] = {}
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0
    
    def fit(self, corpus: iter[Doc]):
        # Computes the data structures with some dark code
        pass

    def tf_function(self, term: int, doc: int) -> float:
        # Abstract method
        raise NotImplementedError()
    
    def _tf_m(self, term: int, doc: int) -> float: 
        # An example implementation of tf_function
        return self.tf[term][doc] / self.doc_max_tf[doc]
    
    # Other several methods to compute tf(t,d)...

    def idf_function(self, term:int) -> float:
        # Abstract method
        raise NotImplementedError()
    
    def _idf_t(self, term: int, a: float = 0):
        # An example implementation of idf_function
        return np.log10((self.corpus_size + a) / (self.df[term] + a)) + a

    # Other several methods to compute idf(t)...

    def dscore(self, term: int, doc: int) -> float:
        # Abstract method
        # Replace with the weight definition of the model
        raise NotImplementedError()

    def retrieve_query(self, query):
        # Abstract method
        # Select the retrieval(s) strategies
        raise NotImplementedError()
 
	def retrieve_feedback(self, query_id: int, rdocs:                list[str], alldocs: list[str]):
      # Abstract method
      # Select the retrieval(s) strategies  in the feedback
        raise NotImplementedError()
     
    def taat_kretrieve(self, query: Query, k: int) -> list[(float, int)]:
        # A strategy for retrieval
        pass

   def daat_kretrieve(self, wquery: dict[int, float], k: int) -> list[ScoredDoc]:
    # A strategy for retrieval in vector space model
    # returns the k documents with the highest score
    	pass

	def daat_tretrieve(self, wquery: dict[int, float], t: float) -> list[ScoredDoc]:
   # A strategy for retrieval in vector space model
   # returns documents with a score greater than a number t
    	pass
   def daat_kretrieve_okapi(self, query: Query, k: int, mode = 0, VR = None, Vdoc = None) -> list[ScoredDoc]:
   # A strategy for retrieval in okapi model
   # returns the k documents with the highest score
   		pass
  def daat_tretrieve_okapi(self, query: Query, t: float, mode = 0, VR = None, Vdoc = None) -> list[ScoredDoc]:
   # A strategy for retrieval in okapi model
   # returns documents with a score greater than a number t
  		pass
```

The `FrequencyModel` class provides:

1. Data structures for the framework.
2. `Fit` method to compute the data structures.
3. Multiple implementations of $df$ and $idf$ functions.
4. Several strategies of retrieval.

Lets present an example on how to build a model with this base.

```python
# src/vs_model.py

class ClassVectorSpaceModel(FrequencyModel):

    def dscore(self, term, doc):
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
    
    def tf_function(self, term, doc):
        return self._tf_m(term, doc)
    
    def idf_function(self, term):
        return self._idf_t(term)

     def retrieve_query(self, query: Query):
        query_id = query.id
        # replacing query words for terms to not repeat 		this step in the loop
        query = [self.vocabulary[word] for word in query if 		word in self.vocabulary]
        if len(query) == 0:
            return []
        query_tf = Counter(query) 
        wquery = self.wquery(query_tf)
        return [
            ScoredDoc(query_id, str(doc), score)
            for score, doc in self.daat_kretrieve(wquery, 15)
        ]
    
    def retrieve_feedback(self, query: Query, rdocs: 		list[str] , alldocs : list[str]):
        wquery = self.feedback(query, rdocs, alldocs)
        
        return [
            ScoredDoc(query.id, str(doc), score)
            for score, doc in self.daat_kretrieve(wquery, 15)
        ]
     def feedback(self, query: Query, rdocs : list[str] , alldocs : list[str]):
            pass
    
```

This model is the vectorial model studied in class lessons:
1. In the `dscore` method, the weight of a term in a document is calculated.  $\frac{tf(t,d) \times idf(t)}{L_d}$.
2. The `wdoc` and `wquery` are methods to calculate the document vector and the query vector respectively.
3. Set the `tf_function` method to return the implementation method `_tf_m`, this method computes $\frac{tf(t,d)}{\max_t tf(t,d)}$. 
4. Set the `idf_function` to return the implementation method `_idf_t`, this method computes $log(\frac{N}{df(t)})$.
5. Set the retrieval strategy to *Document-At-A-Time* with the method `daat_kretrieve` and return the top 15 documents. This documents are represented with `ScoredDoc(query_id, doc_id, score)` from `ir_measures` module. 
6. The `retrieve_feedback` method is called to, from the set of retrieved documents and the relevant ones marked by the user, transform the initial query vector and refine the set of retrieved documents.
7. The new query vector is calculated in the `feedback` method.

The other Model is Okapi:

```python
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

```

This model is a probabilistic model :

1. In the `wt` and `wt_f`,  are methods to calculate the weight of a term in a document. `wt_f` is in the feedback process.

2. Set the `tf_function` method to return the implementation method `_tf_n`, this method computes the frequency of the term.

3. Set the `idf_function` to return the implementation method `_idf_p`, this method computes $log(1 + \frac{N + 0.5}{n_i + 0.5})$.

4. Set the retrieval strategy to *Document-At-A-Time* with the method `daat_kretrieve_okapi` and return the top 15 documents. This documents are represented with `ScoredDoc(query_id, doc_id, score)` from `ir_measures` module. 

5. The `retrieve_feedback` method is called to, from the set of retrieved documents and the relevant ones marked by the user, change the weight vectors of the documents.

   


### 2.4 Evaluation

Evaluation is performed using the module `ir_measures`, this module provides a simple way to test all studied metrics and some extra ones.

Lets build a pipeline for `Cranfield` dataset.

```python
# src/systems/ir_cranfield.py

cranfield = ir_datasets.load('cranfield') # load dataset

class CranfieldIR(IRSystem):
    """Information Retrieval system for Cranfield corpus
    """

    def docs_iter(self):
        for doc in cranfield.docs_iter():
            # format the docs
            yield Doc(int(doc.doc_id), self.text_processor.process(doc.text))

    def retrieve(self, query):
        # format the query and retrieve
        query = Query(query.query_id, self.text_processor.process(query.text))
        return self.model.retrieve(query)
    
    def eval(self, measures):
        # run the system with all queries and store the scored dos
        run = []
        for query in cranfield.queries_iter():
            for scored_doc in self.retrieve(query):
                run.append(scored_doc)
        
        results = ir_measures.calc_aggregate(measures, cranfield.qrels_iter(), run)
        return results
```

With the pipeline ready we can test the system.

```python
# src/main.py

model = ClassVectorSpaceModel()
text_processor = SimpleTextProcessor()
irsystem = CranfieldIR(model, text_processor)
results = irsystem.eval([Success@10, Success@5, P@10, P@5])
print(results)
```

A list of all measures availabe is [here](https://ir-measur.es/en/latest/measures.html#inst)

The Vaswani corpus is evaluated in the same way.

### 2.5. Natural Language Processing

We need a `TextProcessor` to get the words in text and queries so we provide a basic implementation of a processor in the file `nlp.py` however you can provide your own implementation just by inhereting the interface `TextProcessor`.

## 3. Directory structure

```
src/
┣ models/
┃ ┣ __init__.py
┃ ┣ obm25_model.py -> -> Okapi model
┃ ┣ tf_model.py -> FrequencyModel class
┃ ┣ vs_model.py -> Vector space model
┣ systems/
┃ ┣ __init__.py
┃ ┣ irsystem.py -> IRS Pipeline definition
┃ ┣ ir_vaswani.py -> IRS for Vaswani corpus
┃ ┗ ir_cranfield.py -> IRS for Cranfield corpus
┣ tests/ -> Pytests scripts for testing datastructures
┃ ┣ __init__.py
┃ ┣ test_kmax_heap.py
┃ ┗ utils.py
┣ __init__.py 
┣ main.py -> Test the systems
┣ nlp.py -> TextProcessor
┗ utils.py -> Helpful data structures and definitions

```




