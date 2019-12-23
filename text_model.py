# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:49:47 2017

@author: JP

Latent Dirichlet Allocation

"""

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import glob

from time import time

import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim

path = "./movie_reviews/test/*.txt"

files = glob.glob(path)
doc_complete = []
# iterate over the list getting each file 
for fle in files:
   # open the file and then call .read() to get the text 
   with open(fle) as f:
        text = f.read()
        doc_complete.append(text)

#==============================================================================
# Cleaning and Preprocessing
# 
# Cleaning is an important step before any text mining task, in this step, 
# we will remove the punctuations, stopwords and normalize the corpus.
#==============================================================================


stop = set(stopwords.words('english'))
stop |= set(['like',"it's", 'get', "don't", 'even', "you've", 
             "you're", "what's", "didn't", "wasn't"])
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i.strip() not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    short_free = " ".join(w for w in punc_free.split() if len(w) > 4)
    normalized = " ".join(lemma.lemmatize(word) for word in short_free.split())
    return normalized

text_list = [clean(doc).split() for doc in doc_complete]    

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(text_list)
dictionary.save('dictionary.dict')

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)


start = time()
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
numTopix=5
ldamodel = Lda(doc_term_matrix, num_topics=numTopix, id2word = dictionary, passes=50)

print( '\nused: {:.2f}s'.format(time()-start), "\n")

#print(ldamodel.print_topics(num_topics=2, num_words=4))

#for i in ldamodel.print_topics(): 
#    for j in i: print(j)
    

#save model for future use
ldamodel.save('topic.model')

#load saved model
loading = LdaModel.load('topic.model')

#print(loading.print_topics(num_topics=2, num_words=4))


#pyLDAvis.enable_notebook()

d = gensim.corpora.Dictionary.load('dictionary.dict')
c = gensim.corpora.MmCorpus('corpus.mm')
lda = gensim.models.LdaModel.load('topic.model')

data = pyLDAvis.gensim.prepare(lda, c, d)

pyLDAvis.save_html(data,'vis.html')

print('done')







