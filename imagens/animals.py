import numpy as np
import tensorflow as tf
import tempfile
import zipfile
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.preprocessing import LabelBinarizer

temp_dir = tempfile.TemporaryDirectory()
print(temp_dir)
with zipfile.ZipFile('archive.zip', 'r') as zip:
  zip.extractall(temp_dir.name)
print(temp_dir.name)

def load_images_from_main_folder(main_folder):
    images = []
    labels = []

    # Listar todas as subpastas dentro da pasta principal
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

    for folder in subfolders:
        label = os.path.basename(folder)  # Nome da subpasta é o rótulo
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    img_path = os.path.join(folder, filename)
                    img = Image.open(img_path)
                    img = img.convert('RGB')  # Garantir que a imagem esteja no formato RGB
                    img = img.resize((128, 128))  # Redimensionar imagem se necessário
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f'Erro ao processar a imagem {filename} em {folder}: {e}')

    return images, labels

# Especificar a pasta principal que contém as subpastas
main_folder = '/tmp/tmp7ojcdzmp/raw-img'

# Carregar imagens e rótulos das subpastas
images, labels = load_images_from_main_folder(main_folder)

print(f'Número de imagens carregadas: {len(images)}')
print(f'Número de rótulos carregados: {len(labels)}')

if len(images) == 0:
    raise ValueError('Nenhuma imagem carregada. Verifique o caminho das pastas e os arquivos.')

# Codificar rótulos para one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Dividir dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def resize_images(images, target_size):
    resized_images = []
    for img_array in images:
        img = Image.fromarray(img_array)
        img = img.resize(target_size)
        resized_images.append(np.array(img) / 255.0)  # Normalizar aqui
    return np.array(resized_images)

X_train_resized = resize_images(X_train, (64, 64))
X_test_resized = resize_images(X_test, (64, 64))

gerador_treinamento = ImageDataGenerator(rescale=1./255)
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow(X_train_resized, y_train, batch_size=32)
base_teste = gerador_teste.flow(X_test_resized, y_test, batch_size=32)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))

classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classificador.fit(base_treinamento, epochs=10, validation_data=base_teste)

def prepare_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)  # Redimensionar para (64, 64)
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão para o batch
    return img_array

# Caminho da imagem que você quer classificar
image_path = '/tmp/tmp7ojcdzmp/raw-img/mucca/OIP--ZdrmBDHcdX18btrdSdI_wHaEY.jpeg'

# Preparar a imagem
image = prepare_image(image_path, target_size=(64, 64))

# Fazer a predição
predictions = classificador.predict(image)

# Interpretar os resultados
predicted_class = np.argmax(predictions, axis=1)

# Mapear o índice para a classe correspondente
classes = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
predicted_label = classes[predicted_class[0]]

print(f'A imagem foi classificada como: {predicted_label}')
