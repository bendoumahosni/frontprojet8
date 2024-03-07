#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter.ttk import Style
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
import shap
import numpy as np
from joblib import load
from streamlit_shap import st_shap
from datetime import datetime, timedelta
import plotly.express as px
######################################
# Page configuration

st.set_page_config(
    page_title="Prediction Dashboard",
    layout="wide",
    )


df=pd.read_csv('test_app.csv')
list_clients=df['SK_ID_CURR']

ac,imp, inf ,uni_biv = st.tabs(["Accueil","Importance des caracteristiques","Informations client","Graphique d'analyse univariée/bivariée"])
with ac:

    # Découpage de la page web en conteneurs (horizontaux)
    titre = st.container()
    prediction = st.container()

    with titre:
        st.markdown("<h1 style='text-align: center;'>Prédiction de la classe d'un client</h1>", unsafe_allow_html=True)

    with prediction:
        col1,col2 =st.columns([1.5,3.5])
        with col1:
            st.title("resultat classification")    
            def set_gauge(client_id):
                score = prediction_result['predicted_score']
                result= prediction_result['predicted_class']   
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},  
                    title={'text': f"Classe: {result}", 'font': {'size': 45}},
                    delta={'reference': 0.499, 'increasing': {'color': "rgb(255,0,81)"}, 'decreasing': {'color': "rgb(0,139,251)"}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "rgb(49,51,63)"},
                        'bar': {'color': "rgb(49,51,63)"},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'borderwidth': 1,
                        'bordercolor': "black",
                        'steps': [
                            {'range': [0, 0.5], 'color': 'rgb(0,139,251)'},
                            {'range': [0.5, 1], 'color': 'rgb(255,0,81)'}]}))
                return fig_gauge
            client_id = st.selectbox("selectionner le code d'un client",(list_clients))
            st.write('vous avez choisi : ', client_id)
            st.write("le seuil de la classification est : ",0.499)

            # Bouton pour effectuer la prédiction
            if st.button("Effectuer la prédiction"):
            # Effectuer une requête GET à l'API FastAPI
                api_url = f"http://127.0.0.1:8000/predict/{client_id}"
                response = requests.get(api_url)
        
                if response.status_code == 200:
                    prediction_result = response.json()
                    st.success(f"Classe prédite  : {prediction_result['predicted_class']}")
                    st.success(f"Score prédit    : {prediction_result['predicted_score']}")
                    if prediction_result['predicted_class'] == 0:
                        st.markdown('<p style="color:green;font-size: 50px;">Crédit accepté</p>', unsafe_allow_html=True)  
                    else:
                        st.markdown('<p style="color:red;font-size: 50px;">Crédit refusé</p>', unsafe_allow_html=True)
            
                else:
                    st.error(f"Erreur lors de la requête à l'API. Code d'erreur : {response.status_code}")
            
                with col2:
                    st.markdown("<h1 style='text-align: center;color: red;'>Jauge de classification</h1>", unsafe_allow_html=True)
                    col2.plotly_chart(set_gauge(client_id),use_container_width=True)
with inf:
    st.title("Informations du client")
    url=f"http://127.0.0.1:8000/get_info/{client_id}"
    response_info=requests.get(url).json()
    st.write('Client_id :',response_info['client_id'])
    st.write('EXt_SOURCE_1 :',response_info['ext_source_1'])
    st.write('EXt_SOURCE_2 :',response_info['ext_source_2'])
    st.write('EXt_SOURCE_3 :',response_info['ext_source_3'])
    st.write('AMT_GOOD_PRICE :',response_info['good_price'])
    st.write('AMT_ANNUITY :',response_info['amt_annuity'])
    st.write('PAYMENT_RATE :',response_info['payment_rate'])
    
    date_reference = datetime.now()
    duree_en_jours = response_info['days_birth']  
    date_resultat = date_reference + timedelta(days=duree_en_jours)
    st.write('DATE_BIRTH :',date_resultat)
    st.write('GENDER :',"F" if response_info['code_gender']==1 else "M")
    st.write('AMT_CREDit :',response_info['amt_credit'])
    duree_empl = response_info['days_employed']  
    date_emb = date_reference + timedelta(days=duree_empl)
    st.write('DATE_EMBAUCHE :',date_emb)

with imp:
    st.markdown("<h1 style='text-align: center;color: red;'>Importance globale</h1>", unsafe_allow_html=True)

    model = load('lgbm_w.joblib')
    # SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(df.drop(columns=['SK_ID_CURR']))

    # Afficher un summary plot
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, df.drop(columns=['SK_ID_CURR']), show=True)
    st.pyplot(fig_summary)

    st.markdown('Force Plot :', unsafe_allow_html=True)
    # Display force plot for multiple instances
    X=df[df['SK_ID_CURR']==client_id].drop(columns=['SK_ID_CURR'])
    shap_values_l = explainer.shap_values(X)
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values_l[0], X), height=400, width=1000)


    st.title("")
    X=df.drop(columns=['SK_ID_CURR'])
    shap_values = explainer.shap_values(X)
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], X), height=400, width=1000)
       
    ###########################################
    explainer = shap.explainers.Tree(model)
    shap_values = explainer(X)
    st_shap(shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],shap_values[0].values[:,0],
                                        feature_names=X.columns), height=400, width=1000)
    #############################################
    shap_values_explaination = shap.Explanation(shap_values[0], feature_names=X.columns.tolist()) 
    shap.plots.heatmap(shap_values_explaination)
    

with uni_biv:
    uva=st.container()
    bva=st.container()
    with uva:
        df['classe']=model.predict(df.drop(columns=['SK_ID_CURR'])) > 0.499
        df['classe'] = df['classe'].astype(int)
        colors = {'0':'blue', '1':'red' }
        varibale = st.selectbox("Selectionner une variable: ",df.drop(columns=['SK_ID_CURR']).columns)
        def plot_distribution(feature, client_id):
            fig = px.histogram(df, x=feature,color= df['classe'] , color_discrete_map={0: 'blue', 1: 'red'}, marginal='box', 
                        hover_data=['SK_ID_CURR'], nbins=30,
                        title=f"Distribution de {feature} par classe") # color=predicted_classes,
            fig.update_traces(marker=dict(color='rgba(0, 0, 0, 0)', size=10, line=dict(color='rgb(0, 0, 0)', width=2)),
                    selector=dict(mode='markers'))
            fig.add_vline(x=df.loc[df['SK_ID_CURR'] == client_id, feature].iloc[0],
                    line=dict(color="red", width=2),
                    annotation_text=f"Client {client_id}",
                    annotation_position="top left")
            
            return fig 
        if varibale:
            st.plotly_chart(plot_distribution(varibale,client_id))
            
    with bva:    
        variable1 = st.selectbox("Selectionner une premiere variable: ",df.drop(columns=['SK_ID_CURR']).columns)
        variable2 = st.selectbox("Selectionner une deuxieme variable: ",df.drop(columns=['SK_ID_CURR']).columns)
        def plot_bivariate(feature1, feature2, client_id):
            fig = px.scatter(df, x=feature1, y=feature2, 
                            hover_data=['SK_ID_CURR'],color= df['classe'] , color_discrete_map={0: 'blue', 1: 'red'},
                            title=f"Analyse bivariée entre {feature1} et {feature2}")
            fig.add_trace(go.Scatter(x=[df.loc[df['SK_ID_CURR'] == client_id, feature1].iloc[0]],
                                    y=[df.loc[df['SK_ID_CURR'] == client_id, feature2].iloc[0]],
                                    mode="markers",
                                    marker=dict(color="red", size=10),
                                    name=f"Client {client_id}"))
            
            st.plotly_chart(fig)
        if variable1:
            if variable2:
                plot_bivariate(variable1,variable2,client_id)

    
    

    
    
    