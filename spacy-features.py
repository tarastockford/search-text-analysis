#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:43:14 2018

@author: tarastockford
"""

# pip install spacy
# python -m spacy download en_core_web_sm

import pandas as pd
import spacy


# Show all columns in IPython (but Spyder console width can't be changed)
pd.options.display.max_columns = None


# Load English tokenizer, tagger, parser, NER and word vectors
# Other available models: https://spacy.io/usage/models
nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load('en_core_web_md')

# Read CSV
df = pd.read_csv('data/search-console-queries.csv', usecols=['Query'], nrows=100)


#https://stackoverflow.com/questions/46981137/tokenizing-using-pandas-and-spacy
df['nlp'] = df['Query'].apply(lambda x: nlp(x))

# token for token in row


# Find named entities
# https://spacy.io/api/annotation#named-entities
#df['entity'] = df['nlp'].apply(lambda row: [w.text for w in row.ents])
#df['entitylabel'] = df['nlp'].apply(lambda row: [w.label_ for w in row.ents])

# Tried setting to title case first, finds more entities but too many false positives
#df['entity'] = df['Query'].str.title().apply(lambda row: [w.text for w in nlp(row).ents])
#df['entitylabel'] = df['Query'].str.title().apply(lambda row: [w.label_ for w in nlp(row).ents])


# Lemmatize
df['lemma'] = df['nlp'].apply(lambda row: [w.lemma_ for w in row])


# Parts of speech
# https://spacy.io/api/annotation#pos-tagging
df['partsofspeech'] = df['nlp'].apply(lambda row: [w.pos_ for w in row])
df['nounchunks'] = df['nlp'].apply(lambda row: [w.text for w in row.noun_chunks])
df['nounchunkslemma'] = df['nlp'].apply(lambda row: [w.lemma_ for w in row.noun_chunks])
# Matching w.pos as hash values is probably faster than w.pos_ as strings
# ADJ is 83, NOUN is 91, VERB is 99
df['verbs'] = df['nlp'].apply(lambda row: [w.lemma_ for w in row if w.pos == 99])
df['adjectives'] = df['nlp'].apply(lambda row: [w.lemma_ for w in row if w.pos == 83])
df['nouns'] = df['nlp'].apply(lambda row: [w.lemma_ for w in row if w.pos == 91])


# Types of words
df['stopwords'] = df['nlp'].apply(lambda row: [w.text for w in row if w.is_stop])
# df['alphabetic'] = df['nlp'].apply(lambda row: [w.text for w in row if w.is_alpha])
df['digits'] = df['nlp'].apply(lambda row: [w.text for w in row if w.is_digit])
# df['punctuation'] = df['nlp'].apply(lambda row: [w.text for w in row if w.is_punct])
df['likeanumber'] = df['nlp'].apply(lambda row: [w.text for w in row if w.like_num])
df['likeaurl'] = df['nlp'].apply(lambda row: [w.text for w in row if w.like_url])
#df['likeanemail'] = df['nlp'].apply(lambda row: [w.text for w in row if w.like_email])
# df['wordshape'] = df['nlp'].apply(lambda row: [w.shape_ for w in row])

# is_oov doesn't work with en_core_web_sm but does work with the larger models, though not very accurate (some typos missed, some false positives)
#df['outofvocabulary'] = df['nlp'].apply(lambda row: [w.text for w in row if w.is_oov])

