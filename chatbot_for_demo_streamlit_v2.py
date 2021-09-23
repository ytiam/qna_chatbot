#!/usr/bin/env python
# coding: utf-8

import docx2txt
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import glob
from thefuzz import fuzz
import transformers
import tensorflow as tf
from nltk import ngrams
import time
from scipy.spatial import distance
import streamlit as st
import openai
from io import StringIO

openai.api_key = "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"

import os
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS

stp_wrds = list(STOPWORDS)
stp_wrds.extend(['who','how','when','what','which','where','why'])

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np

@st.cache(allow_output_mutation=True,show_spinner=False)
def load_qa_model():
    nlp = pipeline("question-answering",model='distilbert-base-cased-distilled-squad')
    return nlp

@st.cache(allow_output_mutation=True,show_spinner=False)
def load_qus_filtering_model():
    model = SentenceTransformer('stsb-roberta-large')
    return model

@st.cache(allow_output_mutation=True,show_spinner=False,hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def load_docs(files):
    text = []
    for file in files:
        print(file)
        text.append(docx2txt.process(file))
    text_str = ''
    for t in text:
        text_str += t
    text_str = text_str.replace('\n',' ')
    text_str = text_str.replace('\t',' ')
    text_str = text_str.replace('\\x',' ')
    text_str = text_str.replace('.','')
    text_str = text_str.replace(',','')

    text_str_low = text_str.lower()
    return text, text_str, text_str_low

def get_answer(qus):
    ratio_list = []
    for i,t in enumerate(text):
        ratio_list.append(fuzz.token_set_ratio(qus,t))
    max_ratio_list = max(ratio_list)
    indices = []
    for j in range(len(ratio_list)):
        if ratio_list[j] == max_ratio_list:
            indices.append(j)
    filtered_texts = [text[i] for i in indices]
    score_answer = {}
    for t in filtered_texts:
        ans = nlp(question=qus, context=t)
        score_answer[ans['answer']] = ans['score']
    k = max(score_answer, key=score_answer.get)
    v = score_answer[k]
    return k

def get_chatty(prompt):
    start_sequence = "\nAI:"
    restart_sequence = "\nHuman: "

    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      temperature=0.9,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=["\n", " Human:", " AI:"]
    )
    
    r = response['choices'][0]['text']
    save_prompt = prompt+r
    try:
        with open('chat_hist.txt','w') as f:
            f.write(save_prompt)
    except:
        pass
    return r

def conf_prompt(qus):
    with open('chat_hist.txt','r') as f:
        base_prompt = f.read()
    new_prompt = base_prompt + '\nHuman: '+qus+'\nAI:'
    return new_prompt

def get_qus_score(qus):
    qus = qus.lower()
    qus_w_list = qus.split()
    words_to_check = list(set(qus_w_list) - set(stp_wrds))
    text_str_list = [i.strip() for i in text_str_low.split()]
    word_freq_list = []
    for w in words_to_check:
        word_freq_list.append(text_str_list.count(w))
    word_freq_list = [i for i in word_freq_list if i>0]
    if word_freq_list:
        try:
            least_freq = min(word_freq_list)
            score = 1/least_freq
        except ZeroDivisionError:
            score = 0
    else:
        score = 0
    return score

def check_similarity(q1,q2):
    embedding1 = model.encode(q1, convert_to_tensor=True)
    embedding2 = model.encode(q2, convert_to_tensor=True)
    sc = 1 - distance.cosine(embedding1, embedding2)
    return sc

def pass_questions(msg):
    qus = msg.lower()
    faq_qus_list = ['how is the weather today']
    if any([check_similarity(faq,qus)> 0.5 for faq in faq_qus_list]):
        y = True
    else:
        y = False
    return y

def chatbot_response(msg):
    default_gpt3_reply = "Not Relevant Query!!"
    if len(msg.split()) > 2:
        y = pass_questions(msg)
        if y==False:
            score = get_qus_score(msg)
            if score >= 0.2:
                ans = get_answer(msg)
            else:
                ans = default_gpt3_reply #get_chatty(conf_prompt(msg))
        else:
            ans = default_gpt3_reply #get_chatty(conf_prompt(msg))
    else:
        ans = default_gpt3_reply #get_chatty(conf_prompt(msg))
    return ans    

if __name__ == "__main__":
    st.title('Welcome to Smart Chatty...')
    with st.spinner('Question-Answering Model Getting Loaded...'):
        nlp = load_qa_model()
    with st.spinner('Helper Models Getting Loaded...'):
        model = load_qus_filtering_model()
    uploaded_files = st.file_uploader("Drop Multiple Doc Files",accept_multiple_files=True,type=['docx'])
    if uploaded_files:           
        text, text_str, text_str_low = load_docs(uploaded_files)
        st.write('Documents Loaded!')
    question = st.text_input("Ask Question Here")
    button = st.button("Send")
    with st.spinner("Getting Answers.."):
        if button and question:
            ans = chatbot_response(question)
            st.write(f'Answer: {ans}')