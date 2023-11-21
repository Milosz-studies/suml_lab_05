import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os
from transformers.pipelines import pipeline
import asyncio
# zaczynamy od zaimportowania bibliotek

st.set_page_config(
    page_title='Miłosz Dąbrowski - Lab05.',
    page_icon='Image.png'
)

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')


st.title('Miłosz Dąbrowski - Lab05.')
st.image('Image.png')

df = pd.read_csv("DSP_4.csv", sep = ';')
st.dataframe(df)

st.header('Przetwarzanie języka naturalnego')

st.text('Wpisz poniżej tekst w języku angielskim, i wybierz opcję co z nim zrobić')

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie z angielskiego na niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)
elif option == "Tłumaczenie z angielskiego na niemiecki":
    text = st.text_area(label="Wpisz tekst")
    if text:
        with st.spinner(text='Pracuję...'):
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
            translation = translator(text)[0]['translation_text']
            st.success('Done')
        st.write(f'The result:\n{translation}')


st.write('number indeksu: s23010')
