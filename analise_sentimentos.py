import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import csv

resenha = pd.read_csv('imdb-reviews-pt-br - Copia.csv', error_bad_lines=False)
resenha

classificacao = resenha['sentiment'].replace(['neg', 'pos'], [0, 1])
resenha['classificacao'] = classificacao
resenha

#for column in resenha.columns:
#   if resenha[column].dtype == type(object):
#       le = preprocessing.LabelEncoder()
#        resenha[column] = le.fit_transform(resenha[column])

from sklearn.model_selection import train_test_split
treino, teste, classe_treino, classe_teste = train_test_split(resenha.text_pt,
                                                              resenha.classificacao,
                                                              random_state = 45
                                                              
                                                              )
treino = np.array(treino).reshape(len(treino),1)
teste = np.array(teste).reshape(len(teste),1)

regressao_logistica = LogisticRegression()
##regressao_logistica.fit(treino, classe_treino)
##regressao_logistica.score(treino, classe_treino)

"""## selecionar todos os comentários e vetorizá-los. Depois é necessário dividir os dados em treino e teste para então determinar um método de classificação que trabalhe bem com dados esparsos. Por fim, deve-se treinar o classificador escolhido, realizar a previsão e medir os dados de teste."""

vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(resenha.text_pt)
print(bag_of_words.shape)

def classifica_texto(texto, coluna_interesse, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_interesse])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              texto[coluna_classificacao],
                                                              random_state = 45)
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(treino, classe_treino)

print(classifica_texto(resenha, 'text_pt', 'classificacao'))

!pip install WordCloud

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

from wordcloud import WordCloud

palavras = ' '.join([text for text in resenha.text_pt])

palavras_worldcloud = WordCloud(width = 800, height = 500,
                                max_font_size = 110, 
                                collocations = False).generate(palavras) 
                                #determina o tamanho de onde as palavras serao distribuidas, 
                                #nao o tamanho da imagem

import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.imshow(palavras_worldcloud, interpolation='bilinear')
plt.axis('off')
plt.show

def nuvem_palavras_pos(texto, coluna_texto):
  text_pos = texto.query("sentiment == 'pos'")
  palavras = ' '.join([text for text in text_pos[coluna_texto]])

  palavras_wordcloud = WordCloud(width = 800, height = 500,
                                  max_font_size = 110, 
                                  collocations = False).generate(palavras) 
                                  #determina o tamanho de onde as palavras serao distribuidas, 
                                  #nao o tamanho da imagem
  plt.figure(figsize=(10,7))
  plt.imshow(palavras_worldcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show

def nuvem_palavras_neg(texto, coluna_texto):
  text_neg = texto.query("sentiment == 'neg'")
  palavras = ' '.join([text for text in text_neg[coluna_texto]])

  palavras_wordcloud = WordCloud(width = 800, height = 500,
                                  max_font_size = 110, 
                                  collocations = False).generate(palavras) 
                                  #determina o tamanho de onde as palavras serao distribuidas, 
                                  #nao o tamanho da imagem
  plt.figure(figsize=(10,7))
  plt.imshow(palavras_wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show

nuvem_palavras_neg(resenha, 'text_pt')

nuvem_palavras_pos(resenha, 'text_pt')

import nltk
nltk.download('all')

"""### Processo de tokenizacao do nltk"""

frase = ['um filme bom!', 'um filme ruim!']
frquencia = nltk.FreqDist(frase)
frquencia

from nltk import tokenize

frase = 'um filme bom'
token_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_espaco.tokenize(frase)

print(token_frase)

token_frase = token_espaco.tokenize(palavras)
frequencia = nltk.FreqDist(token_frase)
frequencia

df_frequencia = pd.DataFrame({'Palavra': list(frequencia.keys()),
                              'Frequencia': list(frequencia.values())})

df_frequencia

df_frequencia.nlargest(columns = 'Frequencia', n = 10)

import seaborn as sns

def pareto(texto, coluna, quantidade):
  palavras = ' '.join([text for text in texto[coluna]])
  token_frase = token_espaco.tokenize(palavras)
  frequencia = nltk.FreqDist(token_frase)
  df_frequencia = pd.DataFrame({'Palavra': list(frequencia.keys()),
                                'Frequencia': list(frequencia.values())})
  df_frequencia = df_frequencia.nlargest(columns = 'Frequencia', n = quantidade)
  plt.figure(figsize=(12, 8))
  ax = sns.barplot(data = df_frequencia, x = "Palavra", y = "Frequencia")
  ax.set(ylabel = "Contagem")
  plt.show()

pareto(resenha, 'text_pt', 10)

palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')

"""### Tratando o dataframe para tirar as palavras irrelevantes"""

frase_processada = list()
for opiniao in resenha.text_pt:
  nova_frase = list()
  palavras_texto = token_espaco.tokenize(opiniao)
  for palavra in palavras_texto:
    if palavra not in palavras_irrelevantes:
      nova_frase.append(palavra)
  frase_processada.append(' '.join (nova_frase))

frase_processada

resenha['tratamento_1'] = frase_processada
resenha.head()

print(classifica_texto(resenha, 'tratamento_1', 'classificacao'))

"""### Tokenização por ponto"""

from string import punctuation

punctuation

pontuacao = list()

for ponto in punctuation:
  pontuacao.append(ponto)

token_pontuacao = tokenize.WordPunctTokenizer()
pontuacao_stopwords = pontuacao + palavras_irrelevantes


frase_processada = list()
for opiniao in resenha['tratamento_1']:
  nova_frase = list()
  palavras_texto = token_pontuacao.tokenize(opiniao)
  for palavra in palavras_texto:
    if palavra not in pontuacao_stopwords:
      nova_frase.append(palavra)
  frase_processada.append(' '.join(nova_frase))

resenha['tratamento_2'] = frase_processada

print(classifica_texto(resenha, 'tratamento_2', 'classificacao'))

pareto(resenha, 'tratamento_2', 10)

!pip install unidecode
import unidecode 

sem_acento = [unidecode.unidecode(texto) for texto in resenha['tratamento_2']]
sem_acento[0]

stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]

resenha['tratamento_3']  = sem_acento

frase_processada = list()
for opiniao in resenha['tratamento_3']:
  nova_frase = list()
  palavras_texto = token_pontuacao.tokenize(opiniao)
  for palavra in palavras_texto:
    if palavra not in stopwords_sem_acento:
      nova_frase.append(palavra)
  frase_processada.append(' '.join(nova_frase))

resenha['tratamento_3'] = frase_processada

acuracia_tratamento3 = classifica_texto(resenha, 'tratamento_3', 'classificacao')
print(acuracia_tratamento3)

frase_processada = list()
for opiniao in resenha['tratamento_3']:
  nova_frase = list()
  opiniao = opiniao.lower()
  palavras_texto = token_pontuacao.tokenize(opiniao)
  for palavra in palavras_texto:
    if palavra not in stopwords_sem_acento:
      nova_frase.append(palavra)
  frase_processada.append(' '.join(nova_frase))

resenha['tratamento_4'] = frase_processada

acuracia_tratamento4 = classifica_texto(resenha, 'tratamento_4', 'classificacao')
print(acuracia_tratamento4)

stemmer = nltk.RSLPStemmer()

frase_processada = list()
for opiniao in resenha['tratamento_4']:
  nova_frase = list()
  palavras_texto = token_pontuacao.tokenize(opiniao)
  for palavra in palavras_texto:
    if palavra not in stopwords_sem_acento:
      nova_frase.append(stemmer.stem(palavra))
  frase_processada.append(' '.join(nova_frase))

resenha['tratamento_5'] = frase_processada

acuracia_tratamento5 = classifica_texto(resenha, 'tratamento_5', 'classificacao')
print(acuracia_tratamento5)

pareto(resenha, 'tratamento_5', 10)

"""### Uilizando TFIDF para vetorizar palavras com maior peso de relevância"""

from sklearn.feature_extraction.text import TfidfVectorizer

##  exemplo

frase_ex =  ['assisti um filme bom', 'assisti um filme péssimo']

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

caracteristicas = tfidf.fit_transform(frase_ex)

pd.DataFrame(
    caracteristicas.todense(),
    columns = tfidf.get_feature_names()
)

tfidf_bruto  = tfidf.fit_transform(resenha['text_pt'])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_bruto,
                                                              resenha['classificacao'],
                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)

print(acuracia_tfidf)

tfidf_dados_tratados  = tfidf.fit_transform(resenha['tratamento_5'])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_dados_tratados,
                                                              resenha['classificacao'],
                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_tratados = regressao_logistica.score(teste, classe_teste)

print(acuracia_tfidf_tratados)

from nltk import ngrams

tfidf = TfidfVectorizer(lowercase=False, ngram_range=(1,2))
vetor_tfidf = tfidf.fit_transform(resenha['tratamento_5'])

treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                              resenha['classificacao'],
                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)

print(acuracia_tfidf_ngrams)

pesos = pd.DataFrame(regressao_logistica.coef_[0].T,
                     index = tfidf.get_feature_names())

pesos.nlargest(10,0)

pesos.nsmallest(10,0)