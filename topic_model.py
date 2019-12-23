# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:32:22 2017

@author: JP

Latent Dirichlet Allocation
"""

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

#==============================================================================
# Cleaning and Preprocessing
# 
# Cleaning is an important step before any text mining task, in this step, 
# we will remove the punctuations, stopwords and normalize the corpus.
#==============================================================================


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]     

#print(doc_clean,"\n")
'''
OUTPUT
[ ['sugar', 'bad', 'consume', 'sister', 'like', 'sugar', 'father'], 
['father', 'spends', 'lot', 'time', 'driving', 'sister', 'around', 'dance', 'practice'], 
['doctor', 'suggest', 'driving', 'may', 'cause', 'increased', 'stress', 'blood', 'pressure'], 
['sometimes', 'feel', 'pressure', 'perform', 'well', 'school', 'father', 'never', 'seems', 'drive', 'sister', 'better'],
['health', 'expert', 'say', 'sugar', 'good', 'lifestyle'] ] 
'''

#==============================================================================
# Preparing Document-Term Matrix
# 
# All the text documents combined is known as the corpus. To run any mathematical 
# model on text corpus, it is a good practice to convert it into a matrix 
# representation. LDA model looks for repeating term patterns in the entire 
# DT matrix. Python provides many great libraries for text mining practices, 
# “gensim” is one such clean and beautiful library to handle text data. 
# It is scalable, robust and efficient. Following code shows how to convert a 
# corpus into a document-term matrix.
#==============================================================================

#Creating the term dictionary of our courpus, where every unique term is 
#assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

#Converting list of documents (corpus) into Document Term Matrix using 
#dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

#print(doc_term_matrix)
'''
OUTPUT
[ [(0, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)], 
[(3, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1)], 
[(9, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1)], 
[(3, 1), (5, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1)], 
[(0, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1)] ]
'''


#==============================================================================
# Running LDA Model
# 
# Next step is to create an object for LDA model and train it on 
# Document-Term matrix. The training also requires few parameters as input 
# which are explained in the above section. The gensim module allows both 
# LDA model estimation from a training corpus and inference of topic distribution 
# on new, unseen documents.
#==============================================================================

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

#Running and Trainign LDA model on the document term matrix.
#numTpoix gives number of assumed topics found in corpus
numTopix=2
ldamodel = Lda(doc_term_matrix, num_topics=numTopix, id2word = dictionary, passes=50)




#==============================================================================
# Results
#==============================================================================

print(ldamodel.print_topics(num_topics=numTopix, num_words=3))
'''
OUTPUT

['0.168*health + 0.083*sugar + 0.072*bad,
'0.061*consume + 0.050*drive + 0.050*sister,
'0.049*pressure + 0.049*father + 0.049*sister]

Each line is a topic with individual topic terms and weights. 
Topic1 can be termed as Bad Health 
Topic3 can be termed as Family.
'''

#==============================================================================
# Tips to improve results of topic modeling
# 
# The results of topic models are completely dependent on the features (terms) 
# present in the corpus. The corpus is represented as document term matrix, 
# which in general is very sparse in nature. Reducing the dimensionality of the 
# matrix can improve the results of topic modelling. 
# 
# 1. Frequency Filter – Arrange every term according to its frequency. 
# Terms with higher frequencies are more likely to appear in the results as 
# compared ones with low frequency. The low frequency terms are essentially 
# weak features of the corpus, hence it is a good practice to get rid of all 
# those weak features. 
# 
# 2. Part of Speech Tag Filter – POS tag filter is more about the context of the 
# features than frequencies of features. Topic Modelling tries to map out the 
# recurring patterns of terms into topics. However, every term might not be 
# equally important contextually. For example, POS tag IN contain terms such 
# as – “within”, “upon”, “except”. “CD” contains – “one”,”two”, “hundred” etc. 
# “MD” contains “may”, “must” etc. These terms are the supporting words of a 
# language and can be removed by studying their post tags.
# 
# 3. Batch Wise LDA –In order to retrieve most important topic terms, a corpus 
# can be divided into batches of fixed sizes. Running LDA multiple times on 
# these batches will provide different results, however, the best topic terms 
# will be the intersection of all batches.
#==============================================================================
