from pyexpat import model
from models.vs_model import ClassVectorSpaceModel
from systems.ir_vaswani import VaswanidIR
from nlp import SimpleTextProcessor
import ir_datasets
import random

vaswani = ir_datasets.load('vaswani')

tp = SimpleTextProcessor()
vaswani_ir = VaswanidIR(ClassVectorSpaceModel(), tp)
vaswani_model =  vaswani_ir.model
vaswani_model.save('src/models/saved/vaswani.pickle')


def dummy_feedback(model, dataset, rel):
    all_docs = set()
    for doc in dataset.docs_iter():
        all_docs.add(doc.doc_id)

    for query in dataset.queries_iter():
        rel_docs = set()
        qrels = filter(lambda qr: qr.query_id == query.query_id, dataset.qrels_iter())
        for qr in qrels:
            if qr.relevance >= rel:
                rel_docs.add(qr.doc_id)
        for doc in rel_docs:
            for i in range(random.randint(0, 10)):
                model.feedback_on_doc(doc, rel='rel')
        nrel_docs = all_docs.difference(rel_docs)
        for doc in nrel_docs:
            p = random.random()
            if p < 0.6:
                continue
            for i in range(random.randint(0,3)):
                model.feedback_on_doc(doc, rel='nrel')
        

dummy_feedback(vaswani_model, vaswani, 1)
vaswani_model.save('src/models/saved/vaswani_fb.pickle')



    
