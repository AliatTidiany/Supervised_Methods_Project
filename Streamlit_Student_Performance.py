# Importer les packages
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Student_Performance.csv")
pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write("Ce projet s'inscrit dans un contexte scolaire. L'objectif est de prédire la performance académique ou scolaire potentielles d'étudiants. Ce type de jeu de données est souvent utilisé pour analyser et prédire les performances des étudiants en fonction de divers facteurs tels que le temps d'étude, les habitudes de sommeil, les scores précédents, etc. L'objectif peut être de construire un modèle prédictif pour estimer ou prédire l'indice de performance d'un étudiant en fonction de ces paramètres.")
    
    st.write("Nous avons à notre disposition le fichier Student_Performance.csv qui contient des données académiques. Chaque observation en ligne correspond à un étudiant. Chaque variable en colonne est une caractéristique des performances .")
    
    st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire la performance.")
    
    st.image("performance.jpeg")
    
elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    
    st.write("Statistique descriptive", df.describe())
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
    
elif page == pages[2]:
    st.write("### Analyse de données")
    
    fig = sns.displot(x='Performance Index', data=df, kde=True)
    plt.title("Distribution de la variable cible performance")
    st.pyplot(fig)
    
    fig2 = px.scatter(df, x="Performance Index", y="Hours Studied", title="Evolution de la performance en fonction des heures étudiées")
    st.plotly_chart(fig2)
    
    fig3 = px.scatter(df, x="Performance Index", y="Sleep Hours", title="Evolution de la performance en fonction du nombre d'heures de sommeil")
    st.plotly_chart(fig3)
    
    fig4 = px.scatter(df, x="Performance Index", y="Previous Scores", title="Evolution de la performance en fonction des scores obtenus dernièrement")
    st.plotly_chart(fig4)
    
    #Encodage du colonne objet
    from sklearn.preprocessing import LabelEncoder
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
    
    fig5, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    plt.title("Matrice de corrélation des variables du dataframe")
    st.write(fig5)
    

elif page == pages[3]:
    st.write("### Modélisation")
    
    
    
    df_prep = pd.read_csv("df_preprocessed.csv")
    
    Y = df_prep["Performance_Index"]
    X= df_prep.drop("Performance_Index", axis=1)
    
    
    
    #Normaliser les données (X)
    scaler = StandardScaler()
    scaler.fit_transform(X) 
    
    
    #Splitter les données 
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
    #Splitter les données en val et test
    x_val, x_test, y_val, y_test= train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    linear_model = joblib.load('LinearRegression_model.pkl')
    ridge_model = joblib.load('RidgeRegression_model.pkl')
    lasso_model = joblib.load('Lasso_model.pkl')
    
    
    y_pred_lr= linear_model.predict(x_val)
    y_pred_rr=ridge_model.predict(x_val)
    y_pred_lass=lasso_model.predict(x_val)
    
    model_choisi = st.selectbox(label="Modèle", options=['Regression Lineaire', 'Ridge Regression', 'Lasso'])

    def accu(y_true, y_pred):
        R2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return R2, mse

    if model_choisi == 'Regression Lineaire':
    # Effectuer la prédiction avec le modèle de régression linéaire
    # Assumez que y_pred_lr contient les prédictions pour ce modèle
        R2, mse = accu(y_val, y_pred_lr)
        st.write(f"La précision du modèle de régression linéaire est de :  {np.round(R2, 7)}, {np.round(mse, 7)}")

    elif model_choisi == 'Ridge Regression':
    # Effectuer la prédiction avec le modèle de régression Ridge
    # Assumez que y_pred_rr contient les prédictions pour ce modèle
        R2, mse = accu(y_val, y_pred_rr)
        st.write(f"La précision du modèle est de : {np.round(R2, 7)}, {np.round(mse, 7)}")

    elif model_choisi == 'Lasso':
    # Effectuer la prédiction avec le modèle Lasso
    # Assumez que y_pred_lass contient les prédictions pour ce modèle
        R2, mse = accu(y_val, y_pred_lass)
        st.write(f"La précision du modèle Lasso est de :  {np.round(R2, 7)}, {np.round(mse, 7)}")

    
    st.success("La ridge regression est le meilleur modèle d'après les scores obtenus")
    
    params = {
    'alpha': np.logspace(-8, 8, 100)
    }
    models = {
    'LinearRegression': LinearRegression(),
    'RidgeRegression': GridSearchCV(Ridge(), params, cv=5),
    'Lasso': GridSearchCV(Lasso(), params, cv=5)
    }
    # Entraînement du meilleur modèle (Linear Regression dans ce cas)
    best_model = models['RidgeRegression']
    best_model.fit(x_train, y_train)

    # Prédiction sur les données de test (les 3 premières observations)
    x_test_3 = x_test[:3]
    y_pred_3 = best_model.predict(x_test_3)

    # Affichage des prédictions sur Streamlit
    st.write("Affichage des prédictions pour les 3 premiers étudiants :")
    for i in range(3):
        st.write(f'Étudiant {i} - Performance prédite : {np.round(y_pred_3[i], 2)} --- {y_test.iloc[i]} Performance réelle')


    
    

    # Charger le modèle
    model = joblib.load('best_model_ridge_regression.pkl')
    scaler = joblib.load('scaler.pkl')  # Chargez le scaler

# Utilisez le scaler pour normaliser les nouvelles données utilisateur
    # user_data_normalized = scaler.transform(user_data)


    # Interface utilisateur Streamlit
    st.title('Prédiction des performances des étudiants')

    # Ajouter des champs pour saisir les données nécessaires à la prédiction
    hours_studied = st.slider('Nombre d\'heures d\'étude', min_value=1, max_value=9)
    sleep_hours = st.slider('Nombre d\'heures de sommeil', min_value=4, max_value=9)
    previous_scores = st.slider('Scores précédents', min_value=40, max_value=99)
    Extracurricular_Activities = st.slider('Activités extra-scolaire', min_value=0, max_value=1)
    Sample_Question_Papers_Practiced = st.slider('Exemples de questions pratiquées', min_value=0, max_value=9)

    # Bouton pour lancer la prédiction
    if st.button('Prédire'):
        # Organiser les données dans un tableau pour la prédiction
        user_data = np.array([[hours_studied, sleep_hours, previous_scores, Extracurricular_Activities, Sample_Question_Papers_Practiced]])
        
        #scaler = StandardScaler()  # Création de l'objet StandardScaler
        #scaler.fit_transform(x_train, y_train) # Adapter le scaler aux données d'entraînement
        
       
        
        user_data_normalized = scaler.transform(user_data)  # Normaliser les données utilisateur
        
        
    
        
        # Faire la prédiction avec le modèle chargé
        prediction = model.predict(user_data_normalized)
        
        
        
        
       
        
        
        # Afficher la prédiction
        st.write(f"La performance académique prédite de l'étudiant est de: {np.round(prediction[0], 2)}")
        
        

