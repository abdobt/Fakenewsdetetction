# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import graphene
from flask_cors import cross_origin
from graphene import ObjectType, String, Schema
from flask import Flask,request
import json
import NLP
import fakenewsdetection
class Query(ObjectType):
# défintion des champs du query chaque champs prend en argument un string
 stopwords = String(text=String())
 tokeniz=String(text=String())
 lem=String(text=String())
 stem=String(text=String())
 bg=String(text=String())
 fakenewsdetection=String(text=String())
 tf=String(text=String())
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
 def resolve_fakenewsdetection(root,info,text):
   return fakenewsdetection.detect(text)
app = Flask(__name__)
#défintion d'un endpoint
@app.route('/',methods=['POST'])
#paramètres de cross origine
@cross_origin()
def hello_world():
    data = json.loads(request.data)
    schema = Schema(query=Query)
    return json.dumps(schema.execute(" ".join(data['query'].split())).data)
