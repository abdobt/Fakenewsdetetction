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
# l="Hazardous winter conditions delayed the distribution of 6 million doses of coronavirus vaccines this week, the White House announced Friday, hindering lifesaving vaccine drives just as they were gaining momentum. The delayed doses amount to a three-day supply, and this week’s winter storms have slowed the arrival of vaccine in all 50 states, according to Andy Slavitt, the White House senior adviser on the government’s response to the coronavirus. President Biden, speaking at a Pfizer plant in Kalamazoo, Mich., that manufacturers the coronavirus vaccine, said the weather is “slowing up the distribution” of vaccine. News of the magnitude of the slowdown came after thousands of Americans already lost their appointments for second doses because they never arrived or the vaccination site was closed. The ripple effects are expected to stretch into next week as states await delayed shipments and scramble to get their vaccination efforts back on track. At a White House briefing, Slavitt said vaccine shippers — FedEx, UPS and the drug distributor McKesson — “have all faced challenges as workers have been snowed in and unable to get to work.” Road closures in some areas have held up deliveries. And more than 2,000 vaccination sites are in places where power was knocked out. Because the two vaccines allowed for emergency use — manufactured by Pfizer and its German partner BioNTech, and by Moderna — require various degrees of cold storage, it has been important not to risk vaccines arriving in places where scarce doses could be wasted because they could not be properly stored due to storms or power outages. “The vaccines are sitting safe and sound in our factories and hubs,” Slavitt said. Slavitt said the government is asking states and vaccination sites to extend their hours — to reschedule appointments lost because of the storms and to prepare to handle additional vaccine supplies expected in the next weeks and months."
# l = [l]
# j="Dr. Anthony Fauci is staying true to his mission, and is ready to use children as human guinea pigs for coronavirus vaccine experiments, while ignoring the harm the vaccines are doing to a subset of the elderly population already. “Hopefully by the time we get to the late spring, early summer we will have children being able to be vaccinated according to the FDA’s guidance,” said Dr. Anthony Fauci, the nation’s highest paid voice of coercion and control. The only problem with Fauci’s demand: The FDA has no data to justify child exploitation. The drug companies cannot enroll enough adolescents in their experimental trials. All available data shows that the infection is not deadly for children, either.The vaccine companies have struggled to get parental consent in order to use children for the new mRNA vaccine experiments. Moderna publicly admitted that they can’t convince enough parents of children twelve and under to enroll in the experimental trials. “We as an industry cannot just put out a vaccine without testing it,” said Dr. Maria Ryan at the Cottage Hospital in New Hampshire. “And we have to have volunteers to test.” This requires parental consent, something Dr. Fauci would like to bypass.Dr. Fauci’s reckless experiment and desperate ploy to exploit childrenDr. Fauci is recklessly pushing vaccines on children without adequate research in place, without adequate informed consent. He continues to blindly believe that widespread injection of experimental mRNA is the only way to eradicate SARS-CoV-2 and achieve herd immunity. Fauci believes the experimental vaccines will stop viral replication and viral transmission, slowing down the virus’s ability to mutate. On the contrary, medical interventions are known to put pressure on competing pathogens, forcing the virus or bacteria to adapt more rapidly in order to survive among human hosts. Under pressure to survive, pathogens will evolve new traits.As SARS-CoV-2 spread rapidly, it became endemic in most parts of the world (the infection is constantly maintained at a baseline level in a geographic area.) As more immunity is achieved, mortality rates fall. In a way, the virus becomes attenuated (weakened) throughout the population, as human immune systems adapt to their environment. The reality of coronavirus infections is that they are inevitable; the human immune system was designed to adapt to them. Pathogens will always be able to take advantage of de-oxygenated cellular terrains and malnourished immune systems that are suffering from oxidative stress and poor nutrient absorption."
# p="Viruses are always mutating and taking on new forms. The coronavirus has thousands of variants that have been identified. But several, including variants first found in the United Kingdom, South Africa and Brazil, are highly transmissible and have sparked concerns that vaccines may be less effective against them.The same protective measures that have warded off the virus throughout the pandemic — maintaining social distance, wearing masks and washing our hands — are even more critical in the face of more transmissible variants.This mutation, also referred to as 501Y.V2, was found in South Africa in early October and announced in December, when the country’s health minister said the strain seemed to affect young people more than previous strains. This variant may have contributed to a surge of infections and hospitalizations across South Africa.Where is it?This mutation has been identified in more than two dozen countries, including Canada, Australia and Israel. On Jan. 28, South Carolina officials announced that this variant had affected two people there with no travel history — the first instances of this strain identified in the United States. It has since been found in at least nine other states.What makes it different?This mutation shares some similarities to the variant first identified in the U.K. and, like that strain, appears to be more transmissible. There is no evidence that it is more lethal. Scott Gottlieb, former director of the Food and Drug Administration, has suggested that this variant might be more resistant to antibody therapies.There is some evidence that this variant could allow for reinfection: A man in France was in critical condition in mid-February after being infected with this strain four months after he was previously infected with the virus.Will vaccines work?The vaccines may have a diminished impact against this variant, but they probably will still be effective, top infectious-diseases expert Anthony S. Fauci said in January. Moderna has said its vaccine protects against the variant first identified in South Africa, with an important caveat: The vaccine-elicited antibodies were also less effective at neutralizing this mutation in a laboratory dish."
# h="Covid-19 hospitalizations in the United States are at the lowest level since early November, when a fall surge in cases and deaths was picking up steam, data showed Saturday.This comes as federal officials say they're pushing large shipments of vaccines to states this weekend, in part to make up for a backlog from winter storms -- and as public health experts push for faster inoculations before more-transmissible coronavirus variants get a better foothold.About 59,800 Covid-19 patients were in US hospitals on Friday -- down about 55% from a pandemic peak of more than 132,470 on January 6, according to The COVID Tracking Project.Friday's number is the first below 60,000 since November 9, when daily cases, hospitalizations and deaths were on a several-month incline through the holidays.Averages for daily new cases and deaths also have been declining for weeks after hitting all-time peaks around mid-January. Public health experts have been pressing for faster vaccinations, before more transmissible variants have a chance to spread, fearing they could reverse recent progress.The Centers for Disease Control and Prevention has said an apparently more-transmissible variant first identified in the United Kingdom could be the dominant strain in the US by next month.This is why we're telling people to not stop masking, not stop avoiding indoor social gatherings quite yet, because we don't really know what's going to happen with this variant, Dr. Megan Ranney, and emergency medicine physician with Rhode Island's Brown University, told CNN Saturday.And we saw what happened last winter when we didn't take Covid seriously enough.The national test positivity rate -- or the percentage of tests taken that turn out to be positive -- averaged about 4.8% over the last week as of early Saturday, according to The COVID Tracking Project.That's the first time the average has dropped below 5% since October, and it's far below a winter peak of about 13.6% near the start of January. The World Health Organization has recommended governments not reopen until the test positivity rate is 5% or lower for at least two weeks."
# l=[h]
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
#En enlève les caractères spéciaux
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
#En supprime les stopwords
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
#Enregistrement du dataframe dans un fichier real.csv qui sera ajouter à mongodb
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
from wordcloud import WordCloud
real_data = data[data['target'] == 'true']
all_words =  ''.join([text for text in real_data.text])
wordcloud = WordCloud(width= 800, height= 500, max_font_size = 110,
 collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
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
# pipe = Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', LogisticRegression())])
# # Entrainemt du modèle
# model = pipe.fit(X_train, y_train)
# #Prédiction en utilisant les données de test
# prediction = model.predict(X_test)
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
# #calcul de l'accuracy
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# #Calcul de log loss
# scoring = 'neg_log_loss'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
# #Calcul de auc
# scoring = 'roc_auc'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# #Matrice de confusion
# cm = metrics.confusion_matrix(y_test, prediction)
# plot_confusion_matrix(cm, classes=['Fake', 'Real'])
# #Affichge de classification report
# report = classification_report(y_test, prediction)
# print(report)
# print("decision tree classifier")
# from sklearn.tree import DecisionTreeClassifier
# #------------------------------------Decision Tree Classifieer------------------------
# pipe = Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', DecisionTreeClassifier(criterion= 'entropy',
#                                            max_depth = 20,
#                                            splitter='best',
#                                            random_state=42))])
# model = pipe.fit(X_train, y_train)
# #Prédiction en utilisant les données de test
# prediction = model.predict(X_test)
# #calcul de l'accuracy
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# #Calcul de log loss
# scoring = 'neg_log_loss'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
# #Calcul de auc
# scoring = 'roc_auc'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# #Matrice de confusion
# cm = metrics.confusion_matrix(y_test, prediction)
# plot_confusion_matrix(cm, classes=['Fake', 'Real'])
# report = classification_report(y_test, prediction)
# #Affichge de classification report
# print(report)
# print("random forest")
# #---------------------------------Random Forest Classifier------------------------------
# from sklearn.ensemble import RandomForestClassifier
# pipe = Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
# model = pipe.fit(X_train, y_train)
# prediction = model.predict(X_test)
# #calcule de l'accuracy
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# #Calcul de log loss
# scoring = 'neg_log_loss'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
# #Calcul de auc
# scoring = 'roc_auc'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# #Matrice de confusion
# cm = metrics.confusion_matrix(y_test, prediction)
# plot_confusion_matrix(cm, classes=['Fake', 'Real'])
# report = classification_report(y_test, prediction)
# #Affichge de classification report
# print(report)
#Fonction utiliser pour adapter naive bayes et knn au pipeline
class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
# #-----------------------------------------Naive Bayes---------------------
# print("naive bayes")
# pipe = Pipeline([('vectorizer', CountVectorizer ()),
# ('TFIDF', TfidfTransformer ()),
# ('to_dense', DenseTransformer()),
# ('clf', OneVsRestClassifier (GaussianNB()))])
# model = pipe.fit(X_train, y_train)
# prediction = model.predict(X_test)
# # calcul de l'accuracy
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
# #Calcul de log loss
# scoring = 'neg_log_loss'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
# #Calcul de auc
# scoring = 'roc_auc'
# results = model_selection.cross_val_score(model, data['text'], data['target'], cv=kfold, scoring=scoring)
# print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# #Matrice de confusion
# cm = metrics.confusion_matrix(y_test, prediction)
# plot_confusion_matrix(cm, classes=['Fake', 'Real'])
# report = classification_report(y_test, prediction)
# #Affichge de classification report
# print(report)
# print("knn")
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
