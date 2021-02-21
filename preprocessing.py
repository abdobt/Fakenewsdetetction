import nltk
import pandas as pd
import string
import seaborn as sns
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import itertools
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, FunctionTransformer
from nltk import tokenize
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.utils import shuffle
#-------------Importation des datasets
fake = pd.read_csv("fake.csv")
true = pd.read_csv("real.csv")
#-------------Ajout d'une colonne pour classofier les news
fake['target'] = 'fake'
true['target'] = 'true'
#--------------Concaténation des articles réels et fake dans une seule dataframe----------
data = pd.concat([fake, true]).reset_index(drop = True)
#Nous allons mélanger les données pour éviter les biais
data = shuffle(data)
data = data.reset_index(drop=True)
#Supression de la colonne link et id et title
data.drop(["link"],axis=1,inplace=True)
data.drop(["id"],axis=1,inplace=True)
data.drop(["title"],axis=1,inplace=True)
#-----------------Preprocessing------------------
#On enlève les caractères spéciaux
data['text'] = data['text'].str.replace('[#,@,&]', '')
#on conveti les lettres en miniscules
data['text'] = data['text'].apply(lambda x: x.lower())
#fonction pour supprimer les ponctusation
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation and char not in '�']
    clean_str = ''.join(all_list)
    return clean_str
data['text'] = data['text'].apply(punctuation_removal)
#Liste des stopwords(the,a ....)
stop = stopwords.words('english')
#On supprime les stopwords
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#Déclaration d'un toknizer pour faire la tokenization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
#Déclaration d'un lemamatizer
lemmatizer = nltk.stem.WordNetLemmatizer()
#Méthode pour faire la lemmatization
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
# Application de lemmatization
data["text"] = data["text"].apply(lemmatize_text)
#Dtokenization après le stemming et lemmetization
detokenizer = TreebankWordDetokenizer()
data ["text"]= data["text"].apply(detokenizer.detokenize)
#------ Enregistrement du dataframe dans un fichier real.csv qui sera ajouter à mongodb
# data.to_csv('file.csv')
#-----------------Visualisation du dataset---------
#Affichage des premières lignes
print(data.head())
#Nombre de lignes et colonnes
print(data.shape)
#Affichage de resumé statistque
print(data.describe())
# affichage de nombre de nombre de colonnes pour les articles réels et fake
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()
# Affichage de nuage des mots les plus fréquents dans les fakes news
fake_data = data[data['target'] == 'fake']
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,max_font_size = 110,collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# Affichage de nuage des mots les plus utilisés dans les fakes news
# real_data = data[data['target'] == 'true']
# all_words =  ''.join([text for text in real_data.text])
# wordcloud = WordCloud(width= 800, height= 500, max_font_size = 110,
#  collocations = False).generate(all_words)
# plt.figure(figsize=(10,7))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
token_space = tokenize.WhitespaceTokenizer()
#Fonction qui calcule le nombre de fois que un mot est cité et les affiche dans un graphe
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()
# Affichage d'un graphe avec les mots les plus utilisées
counter(data[data["target"] == "fake"], "text", 20)
counter(data[data["target"] == "true"], "text", 20)

#---------------------------------Fonction utilisé pour l'affichage de la matrice de confusion-------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
     plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#Division de données en donnés test et entrainement
X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)
#----------------------------------------Entrainement des modèle---------------------------------------------
#--------------------------------------------Logistic regression------------------------------
print("logistique regression")
#A la place d'effectuer chaque tache séparément on peut passer (l'entrainement,prédiction,transformation.......)
# #On peut une pipeline qui nous permet de passer les trois paramètres (le modèle à utiliser,le transformateur et le compteurs des mots ..)
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])
# Entrainemt du modèle
model = pipe.fit(X_train, y_train)
#Prédiction en utilisant les données de test
prediction = model.predict(X_test)
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
#calcul de l'accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Calcul de log loss
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
#Calcul de auc
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
#Matrice de confusion
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
#Affichge de classification report
report = classification_report(y_test, prediction)
print(report)
print("decision tree classifier")
from sklearn.tree import DecisionTreeClassifier
#------------------------------------Decision Tree Classifieer------------------------
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20,
                                           splitter='best',
                                           random_state=42))])
model = pipe.fit(X_train, y_train)
#Prédiction en utilisant les données de test
prediction = model.predict(X_test)
#calcul de l'accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Calcul de log loss
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
#Calcul de auc
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
#Matrice de confusion
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
report = classification_report(y_test, prediction)
#Affichge de classification report
print(report)
print("random forest")
#---------------------------------Random Forest Classifier------------------------------
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
#calcule de l'accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Calcul de log loss
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
#Calcul de auc
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
#Matrice de confusion
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
report = classification_report(y_test, prediction)
#Affichge de classification report
print(report)
#Fonction utiliser pour adapter naive bayes et knn au pipeline
class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
#-----------------------------------------Naive Bayes---------------------
print("naive bayes")
pipe = Pipeline([('vectorizer', CountVectorizer ()),
('TFIDF', TfidfTransformer ()),
('to_dense', DenseTransformer()),
('clf', OneVsRestClassifier (GaussianNB()))])
model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
# calcul de l'accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Calcul de log loss
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
#Calcul de auc
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
#Matrice de confusion
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
report = classification_report(y_test, prediction)
#Affichge de classification report
print(report)
print("knn")
#---------------------------------------------KNN-------------------------
pipe = Pipeline([('vectorizer', CountVectorizer ()),
('TFIDF', TfidfTransformer ()),
('to_dense', DenseTransformer()),
('clf', OneVsRestClassifier (KNeighborsClassifier(n_neighbors=3)))])
model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
# calcul de l'accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Calcule de log loss
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
#Calcul de auc
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
#Matrice de confusion
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
report = classification_report(y_test, prediction)
#Affichge de classification report
print(report)
