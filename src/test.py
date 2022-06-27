from unittest import result
from systems.ir_vaswani import VaswanidIR
from nlp import SimpleTextProcessor
from models.saved_models import vaswani, vaswani_feedback
from ir_measures import *

tp = SimpleTextProcessor()

system = VaswanidIR(vaswani, tp, fit=False)
result = system.eval([P@5])
print(result)

system = VaswanidIR(vaswani_feedback, tp, fit=False)
print(vaswani_feedback.feedback)
result = system.eval([P@5])
print(result)