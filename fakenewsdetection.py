# -*- coding: utf-8 -*-
import pickle
filename = 'finalized_model.sav'
def detecte(text):
 #On charge le modèle
 loaded_model = pickle.load(open(filename, 'rb'))
 input=[text]
 #On fait un prédiction de l'article
 s = loaded_model.predict(input)
 print(s[0])
 return s[0]