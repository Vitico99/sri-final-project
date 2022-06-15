import nltk
import ir_datasets

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

cranfield = ir_datasets.load("cranfield")
next(cranfield.docs_iter())
