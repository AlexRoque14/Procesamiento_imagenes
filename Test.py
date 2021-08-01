import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 100, 100
modelo = './Model/model.h5'
pesos_modelo = './Model/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)

  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)

  if answer == 0:
    print("pred: Banana")
  elif answer == 1:
    print("pred: Manzana")
  elif answer == 2:
    print("pred: Pera")
  elif answer == 3:
    print("pred: Uva")

print("\n\nPredicción 1: ")
predict('./DATASET/Test2/prueba9.jpg')

