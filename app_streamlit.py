import requests
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import numpy as np

URL = "https://test-projet-v0.herokuapp.com"

st.set_page_config(page_title="Prêt à dépenser", page_icon=":credit_card:", layout="wide")

col1, mid, col2 = st.columns([120,10,25])
with col1:
    st.title("Dashboard : SCORING CREDIT :credit_card:")
with col2:
    st.image('./logo.jpg', width=150)

st.write("Ce dashboard à destination des conseillers permettra d'avoir de manière simple"
" et rapide le score d'octroi d'un crédit pour un client donnée."
" Il contiendra des informations relatives à un client, le score, l'interprétation de ce score "
"ainsi que ses informations comparées à l'ensemble des clients")
st.write("Pour le projet prendre pour exemple les clients suivants '100002', '100011' ou '331040'")

@st.cache_data()
def get_list_clients():
    """API - load list of clients""" 
    response = requests.get(f'{URL}/clients/')
    return list(response.json())

@st.cache_data()
def data_clients():
    """API - load le dataframe clients""" 
    response = requests.get(f'{URL}/client/')
    return response.json()
 
@st.cache_data(show_spinner=False)
def get_data_from_customer(id):
    response = requests.get(f"{URL}/client/{id}")
    return response.json()

def display_customer_data(element):
    if element == '':
        st.subheader(f"Entrer un numéro de clients")
    #elif element not in list_cli:
        #st.subheader(f"Client {element} inconnu: entrer un numéro de clients")
    else:
        r = get_data_from_customer(element)  # on fait appel à l'API
        st.subheader(f"Voici les données et les résultats du client {element}")
        st.write(pd.DataFrame.from_dict(r, orient='index'))
        #element.dataframe(pd.DataFrame.from_dict(r[0]))  # on affiche les données du client

def gauge_plot(probability, threshold):
    value = round(probability, 2) * 100
    percent = 0.1
    # On défini les différents paramètres des sections de la jauge (bornes inf et sup et couleur)
    steps = [([0, max(0, threshold - percent)], (37, 166, 41)),
             ([threshold - percent, max(threshold - percent, threshold + percent)], (235, 159, 27)),
             #([threshold, max(threshold, threshold + percent)], (224, 120, 95)),
             ([threshold + percent, max(threshold + percent, 1)], (242, 40, 26))]

    # On utilise la librairie plotly qui propose des graphiques ultra-paramétrables
    fig = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=value,
            mode="gauge+number+delta",
            title={'text': f""},
            delta={'reference': threshold * 100,
               'increasing': {'color': "rgb(135, 10, 36)"},
               'decreasing': {'color': "rgb(30, 166, 93)"}},
            gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [step[0][0] * 100, step[0][1] * 100], 'color': f'rgb{str(step[1])}'}
                         for step in steps if step[0][0] != step[0][1]],
               'threshold': {'line': {'color': "black", 'width': 8}, 'thickness': 0.75, 'value': threshold * 100}}))

    return fig

@st.cache_data(show_spinner=False)
def get_predict_from_customer(id):
    response = requests.get(f"{URL}/predict/{id}")
    return response.json()

def prediction_cli(element):
    result = get_predict_from_customer(element)
    data_result = pd.DataFrame.from_dict(result, orient='index')
    gauge = gauge_plot(data_result[0]['probability'], data_result[0]['threshold'])
    st.write(gauge)

client = st.sidebar.text_input(label="Saisir l'identifiant d'un client :bust_in_silhouette:")
display_customer_data(client)
prediction_cli(client)