import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv(r"C:/Users/heloi/OneDrive/Área de Trabalho/Curso IA Expert Academy/Redes neurais/dataset_breast_cancer/entradas_breast.csv")
classes = pd.read_csv(r"C:/Users/heloi/OneDrive/Área de Trabalho/Curso IA Expert Academy/Redes neurais/dataset_breast_cancer/saidas_breast.csv")

def criarRede():
    classificador = Sequential()
    #Units qnt neuronios ocultos. Activation, qual forma de ativação. kernel_initializer como será iniciado os pesos iniciais.input_dim é a quantidade de entradas.
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
    classificador.add(Dense(units=1, activation = 'sigmoid'))

    otimizador = keras.optimizers.Adam(learning_rate = 0.001, weight_decay = 0.0001, clipvalue = 0.5)
    #binary_crossentropy fundamentação em regressão logistica
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn=criarRede(), epochs = 100, batch_size = 10)
resultados = cross_val_score(estimator = classificador, X = previsores, y=classes, cv = 10, scoring= 'accuracy')

media = resultados.mean()
desvio = resultados.std()
