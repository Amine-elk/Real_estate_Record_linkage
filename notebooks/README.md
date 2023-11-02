# Requirement 

********************************
Libraries
********************************
pandas
numpy
nltk
matplotlib
seaborn
plotly
sklean
json
re
********************************

### Notebook 0
-------------------------------
function.py : J'ai essayé dans ce notebook d'implémenter les fonctions avec quoi j'ai travaillé.

    'def missing_values_table(df, table=True)': cette fonction prend en arg un dataframe et un booléen, et affiche les valeurs manquantes de chaque colonne dans un diagramme à barres, ainsi qu'un tableau récapitulatif en mettant table = True.

    'drop_func(df, prc, lst_col_keep)' : cette fonction prend en arg un dataframe et un int à partir de quel pourcentage de missing values voulez vous supprimer une colonne, aisni le 3 arg, une list des variables que vous voulez garder meme si elles sont > prc .

    'check_sentence_for_word(sentence, lst)' : cette focntion permet de lister les mots qui sont proches d'une distance de un mot de la list des mots 'lst', et si il est un float, elle return le max, pour notre cas, elle nous permet de returner le chiffre à coté de m², pour la surface.

    'split_measurement_strings(text)' : des fois, on trouve '15m²' au lien de '15 m²', cette fonction regex permet de separer le chiffre de m².

    'preprocess_text_nltk(text)' : cette fonction permet le text processing pour la variable 'DESCRIPTION'.



### Premier notebook
-------------------------------
EDA.ipynb : J'ai essayé dans ce notebook d'annalyser les données (missing values, ...)

### Deuxième notebook
-------------------------------
model_imput_room.ipynb : ce notebook permet de remplir la variable 'ROOM_COUNT' en utilisant un modele de machine learning (RFClassifier).

### Troisième notebook
-------------------------------
Approche.ipynb : ce notebook permet d'identifier les doublons, l'approche que j'ai utilisé pour résoudre ce problème de couplage d'enregistrements.



### raw_data.csv : notre dataset de base ----> sub_data.csv : notre dataset apres room_count filling -----> new_data.csv : notre dataset pour effectuer l'approche
### sub_data.csv : notre dataset apres room_count filling 
### new_data.csv : notre dataset pour effectuer l'approche

#### raw_data.csv  ----> sub_data.csv  -----> new_data.csv 


----------------------------------------------------------------




# Similarity Matrix approach

- select relevant features and distinguish between
	- numeric
	- categorical

- Def scoring functions
	- For catergories
		- scoring function 0-1 score
			APPARTEMNT | BUILDING   ---> 0
			APPARTEMNt | APPARAMENT ---> 1
	- For numerical: 1 - (s_1 - s_2) / (s_1 + s_2)
	- For NA values:
		- if both NA:  ---> 0
		- if one is NA ---> 1

- similarity between offer_1 and offer_2: mean(scores) in [0, 1]

- confidence: (total_number_col - max(NA values)) / total_number_col


- output is a matrix of the form

	offer_1, ...., offer_n
offer_1
   .
   .
   .
offer_n



DID BY : Amine ELKARI