import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.metrics import confusion_matrix, accuracy_score

caminho_arquivo_entrada = r"C:/Users/heloi/OneDrive/Área de Trabalho/Curso IA Expert Academy/Redes neurais/dataset_breast_cancer/entradas_breast.csv"
caminho_arquivo_saida = r"C:/Users/heloi/OneDrive/Área de Trabalho/Curso IA Expert Academy/Redes neurais/dataset_breast_cancer/saidas_breast.csv"
previsores = pd.read_csv(caminho_arquivo_entrada)
classe = pd.read_csv(caminho_arquivo_saida)

#Separação dos dados em 4 variáveis. test_size significa que 25% dos dados então sendo destinados ao teste. Esta utilizando o sklearn.
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

#Para saber a quantidade de neuronios ocultos de partida, soma da qnt de entrada com a possibilidade de saida e divide por 2 (30 + 1) / 2 = 15.5
classificador = Sequential()
#Units qnt neuronios ocultos. Activation, qual forma de ativação. kernel_initializer como será iniciado os pesos iniciais.input_dim é a quantidade de entradas.
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
classificador.add(Dense(units=1, activation = 'sigmoid'))

#binary_crossentropy fundamentação em regressão logistica
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#Treinamento
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 200)


#Avaliação do algoritmo a partir da base de dados teste sklearn de forma manual
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#Avaliaçao do algoritmos a partir da base de dados teste keras
resultado = classificador.evaluate(previsores_teste, classe_teste)
