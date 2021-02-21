import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize, WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
#-------------------Example----------------------
example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""

#------------------Stop words--------------------
#ensemble des stop words en anglais
stop_words = set(stopwords.words('english'))

def stopwords(text):
	#Avant d'enlever les stop words il faut faire la tokenisation
	word_tokens = word_tokenize(text)
	t = [w for w in word_tokens if not w in stop_words]
	return ' '.join(word for word in t)
#----------------------------------------------------
#------------------Tokenization---------------------

def tokenization(text):
	word_tokens = word_tokenize(text)
	return word_tokens
#-----------------------------------------------
#-------------------Lemmatization--------------------
def Lemmatization(text):
	fs = tokenization(text)
	lemmatizer = WordNetLemmatizer()
	for i in np.arange(0,len(fs)):
		fs[i]= lemmatizer.lemmatize(fs[i])
	return fs
#-------------------------------------------
#---------------------Stemming------------------------
def stemming(text):
	sentencestext = []
	#On crée un objet  PorterStemmer
	stemmer = PorterStemmer()
	# tokenization
	words = word_tokenize(text)
	# List comprehension
	words = [stemmer.stem(word) for word in words if word not in stop_words]
	row = ' '.join(words)
	sentencestext.append(row)
	return sentencestext
#------------------------------------
#--------------------BagOfWords-------------------
def BagOfWords(text):
	# Sentence Tokenizer
	s = tokenization(text)
	cv=CountVectorizer()
	res=cv.fit_transform(s).toarray()
	return(res)
#-----------------------TF IDF------------
def TF_IDF(text):
	text=[text]
	tfidf_vectorizer = TfidfVectorizer(use_idf=True)
	tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(text)
	first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
	df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
					  columns=["tfidf"])
	df.sort_values(by=["tfidf"], ascending=False).head(45)
	return df
#----------------------Word2Vec----------------
def  WordToVec(text):
	tokens = word_tokenize(text)
	m = Word2Vec(tokens, min_count=1)
	X = m[m.wv.vocab]
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)
	return result