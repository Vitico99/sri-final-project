from ir_measures import *
from systems.ir_cranfield import CranfieldIR
from obm25_model import OkapiBM25Model
from vs_model import ClassVectorSpaceModel
from nlp import SimpleTextProcessor

model = ClassVectorSpaceModel()
text_processor = SimpleTextProcessor()
irsystem = CranfieldIR(model, text_processor)
results = irsystem.eval([Success @ 10, Success @ 5, P @ 10, P @ 5])
print(results)

model = OkapiBM25Model()
irsystem = CranfieldIR(model, text_processor)
results = irsystem.eval([Success @ 10, Success @ 5, P @ 10, P @ 5])
print(results)
