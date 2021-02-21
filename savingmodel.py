# -*- coding: utf-8 -*-
from pymongo import MongoClient
import pandas
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import pickle
#On se connecte à la base de données
client=MongoClient('mongodb+srv://abdeslam:admin@cluster0.hxu1z.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
db=client.get_database('fakenewsdetection')
#On importe la collection news
data=db.news
#On crée un dataframe
df =  pandas.DataFrame(list(data.find()))
X_train,X_test,y_train,y_test = train_test_split(df['text'], df.target, test_size=0.2, random_state=42)
#----------------------------------------Entrainement des modèle---------------------------------------------
#A la place d'effectuer chaque tache séparément on peut passer (l'entrainement,prédiction,transformation.......)
from sklearn.neighbors import  KNeighborsClassifier
#Fonction utilisé pour adapter knn puisqu'on utilise le pipeline
class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
# On peut une pipeline qui nous permet de passer les trois paramètres (le modèle à utiliser,le transformateur et le compteurs des mots ..)
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', OneVsRestClassifier (KNeighborsClassifier(n_neighbors=3)))])
# Entrainemt du modèle
model = pipe.fit(X_train, y_train)
prediction=model.predict(X_test)
filename = 'finalized_model.sav'
#sauvegarde du modéle
pickle.dump(model, open(filename, 'wb'))