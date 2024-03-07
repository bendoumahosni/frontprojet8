#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter.ttk import Style
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
######################################
# Page configuration
st.set_page_config(
    page_title="Prédiction",
    page_icon="🏂",
    layout="wide",
    )



st.title("Prédiction de classe pour un client")
col1, col2, col3 = st.columns((4, 10,4))

df=pd.read_csv('test_app.csv')
list_clients=df['SK_ID_CURR']
###################################
# premiere colonne 
with col1:
    # Streamlit App
    
    
    client_id = st.selectbox(
        "selectionner le code d'un client",
        (list_clients))

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
#########################################
# Deuxieme colonne                
with col2:
         
    import shap
    import numpy as np
    from joblib import load
    import streamlit.components.v1 as components
    from streamlit_shap import st_shap
    import matplotlib

    model = load('lgbm_w.joblib')
    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(df.drop(columns=['SK_ID_CURR']))

    st.write("### Summary Plot :")
    # Afficher un summary plot
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, df.drop(columns=['SK_ID_CURR']), show=True)
    st.pyplot(fig_summary)

    st.markdown('<p style="color:green;font-size: 50px;text-align: center;">Force Plot :</p>', unsafe_allow_html=True)
    # Display force plot for multiple instances
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], df.drop(columns=['SK_ID_CURR'])), height=400, width=1000)

    ##############################################################
    # les valeurs SHAP 
    st.write('### importance locale ')
    shap.initjs()
    instance=df[df['SK_ID_CURR']==client_id]
    shap_values = explainer.shap_values(instance.drop(columns=['SK_ID_CURR']))
    fig_force, ax_force = plt.subplots()
    shap.force_plot(explainer.expected_value[0], shap_values[0], instance.drop(columns=['SK_ID_CURR']), matplotlib=matplotlib)
    st.pyplot(fig_force)
    #################################################################
    st.write("waterfall")
    shap_values = explainer.shap_values(instance.drop(columns=['SK_ID_CURR']))
    shap.plots.waterfall(shap_values[0])
with col3:
    url=f"http://127.0.0.1:8000/get_info/{client_id}"
    response_info=requests.get(url).json()
    st.write("### Infos client :")
    st.write('Client_id :',response_info['client_id'])
    st.write('EXt_SOURCE_1 :',response_info['ext_source_1'])
    st.write('EXt_SOURCE_2 :',response_info['ext_source_2'])
    st.write('EXt_SOURCE_3 :',response_info['ext_source_3'])
    st.write('AMT_GOOD_PRICE :',response_info['good_price'])
    st.write('AMT_ANNUITY :',response_info['amt_annuity'])
    st.write('PAYMENT_RATE :',response_info['payment_rate'])
    st.write('DAYS_BIRTH :',response_info['days_birth'])
    st.write('CODE_GENDER :',response_info['code_gender'])
    st.write('AMT_CREDit :',response_info['amt_credit'])
    st.write('DAYS_EMPLOYED :',response_info['days_employed'])



# In[ ]:




