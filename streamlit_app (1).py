
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import streamlit as st
from sklearn.metrics import f1_score, accuracy_score
import pickle

df = pd.read_csv("diabetes.csv")

st.dataframe(df.head())

st.sidebar.title("Sommaire🎉")
pages = ["Contexte du projet❄️", "Exploration des données❄️", "Analyse de données❄️", "Modélisation 🎈"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]:
    st.write("### Contexte du projet")

    st.write(" L'ensemble de données provient de l'Institut national du diabète et des maladies digestives et rénales (NIDDK) et vise à résoudre un problème crucial dans le domaine médical : la prédiction diagnostique du diabète. Le diabète est une maladie chronique affectant la régulation du glucose dans le sang et a des implications significatives sur la santé publique.")

    st.write("Nous disposons d'un ensemble de données stocké dans le fichier diabet.csv. Cet ensemble de données comprend plusieurs caractéristiques médicales telles que l'âge, la pression artérielle, le taux de glucose, l'indice de masse corporelle (IMC) et d'autres indicateurs importants. Ces caractéristiques seront utilisées comme variables d'entrée pour le modèle de prédiction.")

    st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire si un patient souffre du diabète ou non .")

    st.image("ldiabete.jpg")

elif page == pages[1]:
    st.write("### Exploration des données")

    st.dataframe(df.head())

    st.write("Dimensions du dataframe :")

    st.write(df.shape)
    st.write("Description des donnees:")
    st.write(df.describe())
    st.write("le nombre d'observation de la variable cible:")
    st.write(df["Outcome"].value_counts())
    if st.checkbox("Afficher les valeurs manquantes"):
        st.dataframe(df.isna().sum())

    if st.checkbox("Afficher les doublons"):
        st.write(df.duplicated().sum())

elif page == pages[2]:
    st.write("### Analyse de données")

    fig3, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax, annot=True)
    plt.title("Matrice de corrélation des variables du dataframe")
    st.write(fig3)

    #with pd.option_context():
    fig = sns.displot(x='Outcome', data=df, kde=True)
    plt.title("Distribution de la variable cible Outcome")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=df, width=0.5, ax=ax, fliersize=10)
    plt.title("visualisation des valeurs aberrantes de chaque variable")
    st.pyplot(fig)

    fig = sns.pairplot(df)
    plt.title("visualisation de la relation entre les variable")
    st.pyplot(fig)

elif page == pages[3]:
    st.write("### Modélisation")

    #df_prep = pd.read_csv("diabetes.csv")

    x = df.drop("Outcome", axis=1).values
    y = df.Outcome.values

    # Normaliser les données
    standard = StandardScaler()
    x= standard.fit_transform(x)
    joblib.dump(standard, 'standard.pkl')

    # spliter les donnees
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    print("C:\\Users\\hp\\Desktop\\ML\\Group 2\\Le-d-ploiement_ML_Streamlit\\model_logisticR:", "model_logisticR.pkl")
    reg = joblib.load("model_logisticR.pkl")
    svm = joblib.load("model_svm.pkl")
    KNN =  joblib.load("neigh.pkl")
    standard= joblib.load('standard.pkl')  

    st.write("Modèles chargés avec succès.")

    y_pred_reg = reg.predict(x_val)
    y_pred_rf = svm.predict(x_val)
    y_pred_kn = KNN.predict(x_val)
    #y_pred_knn = knn.predict(x_val)

    model_choisi = st.selectbox("Modèle", options=['Logistique Regression', 'SVM','KNN'])


    def train_model(model_choisi):
        if model_choisi == 'Logistique Regression':
            y_pred = y_pred_reg
        elif model_choisi == 'SVM':
            y_pred = y_pred_rf
        elif model_choisi== 'KNN':
            y_pred = y_pred_kn  
        f1 = f1_score(y_pred, y_val)
        acc = accuracy_score(y_pred, y_val)
        return f1, acc

    st.write("Le Score F1  et le taux de Précision (accuracy)", train_model(model_choisi))
    st.success("Le KNN est le modèle le plus performant 🎉")
    st.text(" prédictions sur les 10 premières lignes du jeu de test à l'aide du KNN")
# Prédictions
    x_test_3 = x_test[:10]
    y_test_3 = KNN.predict(x_test_3)

# Créer un DataFrame pour les prédictions
    predictions_df = pd.DataFrame({
        'Personne': [f"Personne {i}" for i in range(0, 10)],
        'Statut': ['Diabétique' if status == 1 else 'Non-diabétique' for status in y_test_3]
})

# Afficher le DataFrame dans Streamlit
    st.dataframe(predictions_df)

if page == pages[3]:      
    st.title("Prédiction du Diabète")
    # Champs de saisie pour l'utilisateur
    Pregnancies = st.slider('Nombre de grossesses', min_value=0, max_value=20)
    Glucose = st.slider('Glucose', min_value=0, max_value=199)
    BloodPressure = st.slider('Pression Arterielle', min_value=0, max_value=122)
    SkinThickness = st.slider('Epaisseur de la peau', min_value=0, max_value=99)
    Insulin = st.slider('Insulline', min_value=0, max_value=846)
    BMI = st.number_input('IMC', min_value=0.0, max_value=67.1)
    DiabetesPedigreeFunction = st.number_input('Pourcentage du diabete', min_value=0.0, max_value=2.5)
    Age = st.slider('Âge', min_value=21, max_value=86)
    
    # Créer un tableau NumPy avec toutes les caractéristiques
    user_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Vérifier si l'utilisateur veut faire une prédiction
    if st.button("Faire une prédiction"):
        try:
            # Normaliser les caractéristiques de l'utilisateur avec le même StandardScaler
            user_input_array = standard.transform(user_input)

            # Faire la prédiction avec le modèle de régression logistique
            prediction = KNN.predict(user_input_array)
    
            # Afficher le résultat de la prédiction pour cette personne
            st.write(f"Résultat de la prédiction : {'Diabétique' if prediction == 1 else 'Non-diabétique'}")
        except ValueError as e:
            st.write(f"Erreur lors de la prédiction : {e}")











