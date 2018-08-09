#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:06:51 2018

@author: tarastockford
"""

# pip install -U textblob
# python -m textblob.download_corpora

import pandas as pd
from textblob import TextBlob
from textblob import Word


# Show all columns in IPython (but Spyder console width can't be changed)
pd.options.display.max_columns = None


# Read CSV
df = pd.read_csv('data/search-console-queries.csv', usecols=['Query'], nrows=1000)


df['blob'] = df['Query'].apply(lambda x: TextBlob(x))


# Part-of-speech tagging
df['tags'] = df['blob'].apply(lambda row: [w for w in row.tags])

# Noun phrase extraction
df['nounphrases'] = df['blob'].apply(lambda row: [w for w in row.noun_phrases])

# Lemmatization or stemming
df['lemmas'] = df['blob'].apply(lambda row: [w for w in row.words.lemmatize()])
df['singular'] = df['blob'].apply(lambda row: [w for w in row.words.singularize()])
df['stem'] = df['blob'].apply(lambda row: [w for w in row.words.stem()])


# Word defintions
df['definitions'] = df['blob'].apply(lambda row: [Word(w).definitions for w in row.words])


# Spelling correction (false positives; quite slow)
df['spellcheck'] = df['blob'].apply(lambda row: [w.spellcheck() for w in row.words])
df['corrected'] = df['blob'].apply(lambda row: [w.correct() for w in row.words])


# Language detection (often wrong) and translation
# Powered by Google Translate API
df['language'] = df['blob'].apply(lambda row: [row.detect_language()])
# Translation returns errors if it can't be translated
# df['translated'] = df['blob'].apply(lambda row: [w for w in row.translate(to='en') if row.detect_language() != 'en'])

