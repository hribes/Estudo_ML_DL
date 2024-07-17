#importações

import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

#Adicionando o modelo do TensorFlow e o local da URL

model_name = “efficientnetv2-xl-21k”
model_handle_map = {"efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"}
model_image_size_map = {“efficientnetv2-xl-21k”,512}

#Identificador do modelo que esta sendo utilizado

model_handle = model_handle_map.get(model_name)
pixels = model_image_size_map.get(model_name,224)
print(f"Selecionado o modelo: {model_name}:{model_handle}")

#Corrigindo tamanho da imagem e do lote

IMAGE_SIZE = (pixels,pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 16 #Lote

#Obtendo o conjunto de dados (messe caso as fotos de flores)

data_dir = tf.keras.utils.get_file(
	'flower_photos',
	#Nome do arquivo
	'https://storage.googleapis.com/download.tensorflow.org/examples_images/flower_photos.tgz',
	#A cima é o caminho para encontrar o arquivo das fotos
	untar=True
)

#Com essa função sera feito o processamento de imagem com algumas especificações como caminho, validação, subconjunto, rotulo, semente, tamanho da imagem e tamanho do lote

def build_dataset (subset):
	return tf.keras.preprocessing.iamge_dataset_from_directory(
		data_dir,
		validation_split=.20,
		subset=subset,
		label_mode='categorical',
		seed=123,
		image_size = IMAGE_SIZE,
		batch_size=1
	)

#Cria um conjunto de dados de treino

train_ds = build_dataset('training')
class_name = tuple(train_ds.class_names)
train_size = train_ds.cardinality().numpy()
train_ds = train_ds.unbatch().batch(BATCH_SIZE)
train_ds = train_ds.repeat()

normalization_layer = tf.keras.layers.Rescaling(1./255) #Essa é a criação de uma camada de pré processamento, normalizando os valores de entrada
preprocessing_model = tf.keras.Sequential([normalization_layer]) #Com isso irá criar um modelo sequencial

#Controi um conjunto de dados (usado para ter uma maior variedade de dados de entrada)

do_data_augmentation = False
if do_data_augmentation:
	preprocessing_model.add(
		tf.keras.layers.RandomRotation(40)
	)
	preprocessing_model.add(
		tf.keras.layers.RandomTranslation(0,0.20)
	)
	preprocessing_model.add(
		tf.keras.layers.RandomTranslation(0.2,0)
	)
	preprocessing_model.add(
		tf.keras.layers.RandomZoom(0.2,0.2)
	)
	preprocessing_model.add(
		tf.keras.layers.RandomFlip(mode='horizontal')
	)

train_ds = train_ds.map(lambda images, labels: (preprocessing_model(images), labels))

#Controi um conjunto de dados para validação (utilizará o restante de dados que não foram utilizados no treinamento)

val_ds = build_dataset("validation")
valid_size = val_ds.cardinality().numpy()
val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.map(lambda images, labels: (normalization_layer(images), labels))

#Define o modelo que será usado

do_fine_tuning = False
print("Building model with", model_handle)

model = tf.keras.Sequential([
	tf.keras.layers.InputLayer(input_shape = IMAGE_SIZE + (3,)),
	hub.KerasLayer(model_handle, trainable=do_fine_tuning),
	tf.keras.layers.Dropout(rate=0.2),
	tf.keras.layers.Dense(len(class_names), kernel_regularizer=tf.keras.regularizers.12(0.0001))
])

#Constroi o modelo

model.build((None,) + IMAGE_SIZE+(3,))
model.summary()

#Faz o treinamento do modelo

model.compile(
	optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0,9),
	loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
	metrics=['accuracy']
)

#Faz os ajustes dos modelo

steps_per_epoch = BATCH_SIZE #train_size
validation_steps = BATCH_SIZE #valid_size

history = model.fit(
	tarin_ds,
	epochs=5,
	steps_per_epochs=steps_per_epoch,
	validation_data=val_ds,
	validation_steps=validation_steps
).history

#Constroi grafico apresentando a perda para treinamento e validação

plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Trainin steps")
plt.ylim([0,2])
plt.plot(hist['loss'])
plt.plot(his['val_loss'])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Trainin steps")
plt.ylim([0,1])
plt.plot(hist['Accuracy'])
plt.plot(his['val_accuracy'])

#Obtem os dados para o grafico
x, y = next(iter(val_ds))
image = x[0, :, :, :]
true_indez = np.argmax(y[0])
plt.imshow(image)
plt.axis('off')
plt.show()

prediction_scores = model.predict(np.expand_dims(image,axis=0))
predicted_indez = np.argmax(prediction_scores)
print("True label:" + class_names[true_index])
print("Predicted label:" + class_names[predicted_index])

#Salva o modelo
saved_model_path = f"/tbm/saved_flowers_model_{model_name}"
tf.saved_model.save(model,saved_model_patch)
