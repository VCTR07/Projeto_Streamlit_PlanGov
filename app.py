
import streamlit as st

import pdfplumber
import re
from collections import Counter
from itertools import islice
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from goose3 import Goose
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words('portuguese')

#Integra todas as paginas do documento
def documentoIntegral(planoGov):
    documento=planoGov
    pdf=pdfplumber.open(documento)
    conteudo=''
    for item in pdf.pages:
      conteudo=conteudo+item.extract_text()

    return conteudo

#Remove pontuação
def removerPontuacao(conteudo):
    conteudo=re.sub(r'[^\w\s]',' ', conteudo)
    return conteudo

#Tokeniza
def tokenize (conteudo):
    conteudo=re.findall(r'\w+', conteudo)
    return conteudo

#Remove StopWords
def remover_stopWords(conteudo):
    conteudo_limpo=[]
    for item in conteudo:
      if (item not in stopwords) & (len(item)>1):
        conteudo_limpo.append(item)
    return conteudo_limpo

#Conta e ordena as palavras
def count_sort_n_tokens(conteudo):
    conteudo_limpo1=Counter(conteudo)
    conteudo_limpo1=conteudo_limpo1.most_common(30)
    return conteudo_limpo1

#Pipelines
pipeline_full = [str.lower, removerPontuacao, tokenize, remover_stopWords, count_sort_n_tokens]
pipeline_prep_geral=[str.lower, removerPontuacao, tokenize, remover_stopWords]

#Preparação Geral
def prepare_geral(conteudo, pipeline_prep_geral):    
    tokens = conteudo
    for transform in pipeline_prep_geral:
        tokens = transform(tokens)    
    return tokens

#Preparação Específica
def prepare(conteudo, pipeline=None):
    if pipeline is None:
        pipeline = pipeline_full
    tokens = conteudo
    for transform in pipeline:
        tokens = transform(tokens)
    df = pd.DataFrame(tokens, columns=['Palavras', 'Quantidade'])
    return df

#Grafico tokens
def plot_line(df):
    fig = go.Figure([go.Bar(x=df.Palavras, y=df.Quantidade,
                            text=df.Quantidade,
                            textposition='auto')])
    fig.update_layout(
        autosize=False,
        width=500,
        height=500)
    return fig    

#Cria bigramas
def bigramas(conteudo):    
    
    conteudo_limpo=prepare_geral(conteudo, pipeline_prep_geral)
    bigrams= [*map(' '.join, zip(conteudo_limpo, islice(conteudo_limpo, 2, None)))]
    
    stats_bigrams = Counter(bigrams)
    stats_bigrams = stats_bigrams.most_common(30)
    
    df = pd.DataFrame(stats_bigrams, columns=['Palavras', 'Quantidade'])
    return df

#Grafico Bigramas
def plot_line_bigramas(df):
    fig = go.Figure([go.Bar(x=df.Palavras, y=df.Quantidade,
                            text=df.Quantidade,
                            textposition='auto')])
    fig.update_layout(
        autosize=False,
        width=500,
        height=500)
    return fig   

#Nuvem de palavras
def nuvemPalavras(conteudo):    
    conteudo_limpo=prepare_geral(conteudo, pipeline_prep_geral)
    all_tokens = " ".join(s for s in conteudo_limpo)
    
    
    w = WordCloud().generate(all_tokens)
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3])
    return plt.imshow(w), st.pyplot(fig)


#Escolhe o plano a ser analisado
def escolherCandidato():
    print("Para análise do plano de governo, escolha o seu candidato.")
    print("Digite: 1 - para Bolsonaro; 2 - para Ciro; 3 - para Davila; 4 - para Kelmon; 5 - para Lula; 6 - para Soraya")
    opcao=int(input("Digite um número: "))
    if opcao == 1:
        conteudo = documentoIntegral('plano-de-governo-2023-2026-jair-bolsonaro.pdf')
        return conteudo
    elif opcao == 2:
        conteudo = documentoIntegral('ciro_360.pdf')
        return conteudo
    elif opcao == 3:
        conteudo = documentoIntegral('Plano-de-Governo-Felipe-Davila.pdf')
        return conteudo
    elif opcao == 4:
        conteudo = documentoIntegral('padreKelmon.pdf')
        return conteudo
    elif opcao == 5:
        conteudo = documentoIntegral('plano_gov_lula.pdf')
        return conteudo
    elif opcao == 6:
        conteudo = documentoIntegral('Plano-de-governo-Soraya-Thronicke.pdf')
        return conteudo
    else:
        return "Opção Inválida"

#Inicio do programa
def analisePlano():
    conteudo=escolherCandidato()
    df=prepare(conteudo)
    plot_line(df)
    df=bigramas(conteudo)
    plot_line_bigramas(df)
    #nuvemPalavras(conteudo)

#analisePlano()

#Criando DF com os candidatos e respctivos planos
candidatos=["Bolsonaro", "Ciro", "Daviala", "Kelmon", "Lula", "Soraya"]
planos=[documentoIntegral('plano-de-governo-2023-2026-jair-bolsonaro.pdf'), documentoIntegral('ciro_360.pdf'), documentoIntegral('Plano-de-Governo-Felipe-Davila.pdf'), documentoIntegral('padreKelmon.pdf'), documentoIntegral('plano_gov_lula.pdf'), documentoIntegral('Plano-de-governo-Soraya-Thronicke.pdf')]
lista_de_tuplas = list(zip(candidatos, planos))
df_planos = pd.DataFrame(lista_de_tuplas, columns=['Candidatos', 'Planos'])
#df_planos.Candidatos[0]

st.title('Análise do Plano de Governo dos Candidatos à Presidência da República nas Eleições de 2022.')
st.write('Nesse projeto vamos analisar as palavras mais citadas nos respectivos planos de governo, o que pode indicar as prioridades de cada candidato.')

checkbox_mostrar_candidatos = st.sidebar.checkbox('Mostrar Candidatos')
if checkbox_mostrar_candidatos:

    st.sidebar.markdown('## Seleção de Candidato')

    candidatos = list(df_planos['Candidatos'])
    

    candidato = st.sidebar.selectbox('Selecione o candidato para apresentar respectiva análise do plano de governo.', options = candidatos)

    if candidato == 'Bolsonaro':
        #conteudo=df_planos.Planos[df_planos.index[(df_planos.Candidatos == 'Bolsonaro')]].astype(str)
        conteudo = documentoIntegral('plano-de-governo-2023-2026-jair-bolsonaro.pdf')
        df=prepare(conteudo)
        st.subheader('Palavras mais citadas no plano de governo')
        st.plotly_chart(plot_line(df))
        df=bigramas(conteudo)
        st.subheader('Bigramas mais relevantes no plano de governo')
        st.plotly_chart(plot_line_bigramas(df))
        st.subheader('Nuvem de palavras do plano de governo')
        nuvemPalavras(conteudo)
    elif candidato == 'Ciro':
        #conteudo=df_planos.Planos[df_planos.index[(df_planos.Candidatos == 'Ciro')]]
        conteudo = documentoIntegral('ciro_360.pdf')
        df=prepare(conteudo)
        st.subheader('Palavras mais citadas no plano de governo')
        st.plotly_chart(plot_line(df))
        df=bigramas(conteudo)
        st.subheader('Bigramas mais relevantes no plano de governo')
        st.plotly_chart(plot_line_bigramas(df))
        st.subheader('Nuvem de palavras do plano de governo')
        nuvemPalavras(conteudo)
    elif candidato == 'Daviala':
        #conteudo=df_planos.Planos[df_planos.index[(df_planos.Candidatos == 'Daviala')]]
        conteudo = documentoIntegral('Plano-de-Governo-Felipe-Davila.pdf')
        df=prepare(conteudo)
        st.subheader('Palavras mais citadas no plano de governo')
        st.plotly_chart(plot_line(df))
        df=bigramas(conteudo)
        st.subheader('Bigramas mais relevantes no plano de governo')
        st.plotly_chart(plot_line_bigramas(df))
        st.subheader('Nuvem de palavras do plano de governo')
        nuvemPalavras(conteudo)
    elif candidato == 'Kelmon':
        #conteudo=df_planos.Planos[df_planos.index[(df_planos.Candidatos == 'Kelmon')]]
        conteudo = documentoIntegral('padreKelmon.pdf')
        df=prepare(conteudo)
        st.subheader('Palavras mais citadas no plano de governo')
        st.plotly_chart(plot_line(df))
        df=bigramas(conteudo)
        st.subheader('Bigramas mais relevantes no plano de governo')
        st.plotly_chart(plot_line_bigramas(df))
        st.subheader('Nuvem de palavras do plano de governo')
        nuvemPalavras(conteudo)
    elif candidato == 'Lula':
        #conteudo=df_planos.Planos[df_planos.index[(df_planos.Candidatos == 'Lula')]]
        conteudo = documentoIntegral('plano_gov_lula.pdf')
        df=prepare(conteudo)
        st.subheader('Palavras mais citadas no plano de governo')
        st.plotly_chart(plot_line(df))
        df=bigramas(conteudo)
        st.subheader('Bigramas mais relevantes no plano de governo')
        st.plotly_chart(plot_line_bigramas(df))
        st.subheader('Nuvem de palavras do plano de governo')
        nuvemPalavras(conteudo)
    elif candidato == 'Soraya':
        #conteudo=df_planos.Planos[df_planos.index[(df_planos.Candidatos == 'Soraya')]]
        conteudo = documentoIntegral('Plano-de-governo-Soraya-Thronicke.pdf')
        df=prepare(conteudo)
        st.subheader('Palavras mais citadas no plano de governo')
        st.plotly_chart(plot_line(df))        
        df=bigramas(conteudo)
        st.subheader('Bigramas mais relevantes no plano de governo')
        st.plotly_chart(plot_line_bigramas(df))
        st.subheader('Nuvem de palavras do plano de governo')
        nuvemPalavras(conteudo)  
    



