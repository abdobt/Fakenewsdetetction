# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import graphene
from flask_cors import cross_origin
from graphene import ObjectType, String, Schema
from flask import Flask,request
import json
import NLP
import numpy as np
import fakenewsdetection
class Query(ObjectType):
# défintion des champs du query chaque champs prend en argument un string
 #pour stopwords
 stopwords = String(text=String())
 #pour tokenization
 tokeniz=String(text=String())
 #pour la lemmitization
 lem=String(text=String())
 #pour stemming
 stem=String(text=String())
 #pour bag of words
 bg=String(text=String())
 #pour tf-idf
 tf=String(text=String())
 #pour word2vec
 wordv=String(text=String())
 fakenewsdetection=String(text=String())
    # our Resolver method takes the GraphQL context (root, info) as well as
    # Argument (name) for the Field and returns data for the query Response
 def resolve_stopwords(root, info, text):
   return NLP.stopwords(text)
 def resolve_tokeniz(root, info, text):
   return NLP.tokenization(text)
 def resolve_lem(root,info,text):
   return NLP.Lemmatization(text)
 def resolve_stem(root,info,text):
   return NLP.stemming(text)
 def resolve_bg(root,info,text):
   return NLP.BagOfWords(text)
 def resolve_tf(root,info,text):
   return NLP.TF_IDF(text)
 def resolve_wordv(root,info,text):
   return NLP.WordToVec(text)
 def resolve_fakenewsdetection(root,info,text):
   return fakenewsdetection.detecte(text)
app = Flask(__name__)
#défintion d'un endpoint
@app.route('/',methods=['POST'])
#paramètres de cross origine
@cross_origin()
def hello_world():
    data = json.loads(request.data)
    #Création d'une shéma
    schema = Schema(query=Query)
    #on retourne le resultat sous forme json après l'éxecution
    return json.dumps(schema.execute(" ".join(data['query'].split())).data)