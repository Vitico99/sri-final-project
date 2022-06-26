from termios import FF1
from ir_measures import *
from systems.ir_cranfield import CranfieldIR
from systems.ir_vaswani import VaswanidIR
from models.obm25_model import OkapiBM25Model
from models.vs_model import ClassVectorSpaceModel
from nlp import SimpleTextProcessor
import streamlit as st 
from utils import Query


def init_session():
    st.session_state['model'] = None
    st.session_state['corpus'] = None
    st.session_state['system'] = None
    st.session_state['result'] = []
    st.session_state['relevants'] = {}


def clean_checkboxes():
    for i in range(len(st.session_state['result'])):
        st.session_state[f"rel{i}"] = False


def get_irsystem(model, corpus):
    if st.session_state['model'] == model and st.session_state['corpus'] == corpus:
        return st.session_state['system']

    st.session_state['model'] = model
    st.session_state['corpus'] = corpus

    text_processor = SimpleTextProcessor()
    if model == 'ClassVectorSpace':
        model = ClassVectorSpaceModel()
    else:
        model = OkapiBM25Model()

    if (corpus == "Cranfield"):
        st.session_state['system'] = CranfieldIR(model, text_processor)
    else:
        st.session_state['system'] = VaswanidIR(model, text_processor)
    return st.session_state['system']


if 'system' not in st.session_state:
    init_session()


st.title('Welcome to the Information Retrieval System! 👋')
st.sidebar.header('Introduce required data')

corpus = st.sidebar.radio("Select corpus", ("Cranfield", "Vaswani")) 
model = st.sidebar.radio("Select model", ("ClassVectorSpace", "OkapiBM25")) 
action = st.sidebar.radio("Select mode", ("Retrieve query", "Evaluate model")) 


if action == "Retrieve query":
    st.session_state['text_query'] = st.sidebar.text_input('Introduce query')   
    query_split = st.session_state['text_query'].split()
    query = Query(1, query_split)
    
    if st.sidebar.button('Retrieve'):
        clean_checkboxes()
        irsystem = get_irsystem(model, corpus)
        scores_docs = irsystem.model.retrieve_query(query)
        
        st.session_state['result'] = []
        for item in scores_docs:
            st.session_state['result'].append(irsystem.get_doc(item.doc_id))   

    if st.sidebar.button("Improve"):  
        clean_checkboxes()      
        if len(st.session_state['result']) != 0:
            relevants = []
            for i in range(len(st.session_state['result'])):
                if st.session_state['relevants'][i]:
                    relevants.append(st.session_state['result'][i].doc_id)

            if len(relevants) != 0:
                irsystem = get_irsystem(model, corpus)
                scores_docs = irsystem.model.retrieve_feedback(query, relevants, st.session_state['result'])
            
                st.session_state['result'] = []
                for item in scores_docs:
                    st.session_state['result'].append(irsystem.get_doc(item.doc_id))  

    if 'result' in st.session_state:
        _, col1, col2, colt = st.columns([1.3, 1, 1, 25])
        col1.text('')
        col2.text('R')
        colt.text('Title of document')
        for i, r in enumerate(st.session_state['result']):
            col3, col4, col5, col6 = st.columns([1.3, 1, 1, 25])
            col3.text(f'{i}. ')
            globals()[f"doc{i}"] = col4.checkbox(label = '', key=f"doc{i}")
            st.session_state['relevants'][i] = col5.checkbox(label='', key=f"rel{i}", value=False)

            try:
                col6.text(r.title)
            except:
                col6.text(r.doc_id)
            if globals()[f"doc{i}"]:
                st.markdown(r.text)


if action == "Evaluate model":
    init_session()
    n = st.sidebar.text_input("Introduce number of documents to analize")
    if st.sidebar.button('Evaluate'):
        irsystem = get_irsystem(model, corpus)
        results = irsystem.eval([Success @ int(n), P @ int(n)])
        st.text(results)
