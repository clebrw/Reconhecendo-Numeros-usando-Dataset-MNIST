import cv2                               			# OpenCV
import numpy as np                       			# NumPy

from keras.datasets import mnist         			# Importando o dataset usado no treino
from keras.models import Sequential      			# Modelo de rede neural
from keras.layers import Dense           			# Layer do tipo densamente conectado
from keras.utils import np_utils         			# Usaremos dela o metodo 'to_categorical()'

# Carregando o Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)              # reshape(linhas, colunas)
x_test = x_test.reshape(10000, 784)

x_train = x_train/255.0    
x_test = x_test/255.0	

y_train = np_utils.to_categorical(y_train)         # Transformando lista em matriz
y_test =np_utils.to_categorical(y_test)

# Criando o modelo do tipo Sequencial
model = Sequential() 

# Camada Oculta
model.add(Dense(256, input_dim=784, activation='relu'))       # Adicionando a camada densa. Dense(qtde_de_neurônios, input_dim = qtde_de_entradas, Activation='tipo_de_ativação')
# 784 = qtde_pixel
# 256 = valor arbitrário, altere por valores de potência de 2. Ex: 2, 4, 8, 16, 32, 64...

# Camada de Saída
model.add(Dense(10, activation='softmax'))                    # Adicionando a camada de saída Dense(qtde_de_neuronios, activation='tipo_de_ativação')
# 10 = qtde_classes

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, epochs=5)

evaluate = model.evaluate(x_test, y_test)

print('\nloss:{:3.2f}, accuracy:{:2.2f}'.format(evaluate[0], evaluate[1]))

##### Usando na predição uma imagem do computador, DESCOMENTE SE FOR USÁ-LA
# OBS: a imagem deve ter o fundo preto com o numero em branco!
# img = cv2.imread('numero.png')
# img = cv2.resize(img, (28, 28))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img.reshape(1, 28*28)
# img_do_pc = img/255.0

##### Usando uma imagem do dataset de testes 
img_do_dataset = x_test[0].reshape(1, 28*28)

##### Prevendo o valor
# Use img_do_pc ou img_do_dataset
resultado = model.predict(img_do_dataset)

print('Valor previsto: ',resultado.argmax())
print('Precisão: {:4.2f}%'.format(resultado.max()* 100))