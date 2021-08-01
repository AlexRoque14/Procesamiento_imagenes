import itertools
import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras.preprocessing import image
from mlxtend.evaluate import confusion_matrix

K.clear_session()
print("Execution Training...")

data_training = './DATASET/Train'                   #Directorio del dataset train
data_validation = './DATASET/Validation'            #Directorio del dataset validation


IMAGE_WIDTH = 100                                     #Ancho de la imagen.
IMAGE_HEIGHT = 100                                    #Altura de la imagen.
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)              #Tamaño de la imagen.
IMAGE_CHANNELS=3                                    #Colores RGB.
CLASSES = 4                                         #Número de clases (manzana, pera, uva , banana).

epoch = 50                                          #Numero de ciclos de entrenamiento.
batch = 13                                          #Imagenes que recibirá. barch x steps = aprox de imagenes que se tiene.
batch_size = 1                                     #32 es el batch size por default

Steps = 10                                          #Cuantas veces se va procesar la info en una epoca.
learn_rate = 0.005                                  #tasa de aprendizaje.


#Pre procesamiento de las imagenes ------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale = 1. /255 ,                             #Todos los pixeles van de 0 a 1 para mas eficiencia.
    shear_range = 0.3 ,                             #Inclina las imagenes.
    zoom_range = 0.3 ,                              #hace zoom a algunas imagenes.
    horizontal_flip = True                          #invierte la imagen.
)

#Reescala las imagenes de validación
validation_datagen = ImageDataGenerator(
    rescale = 1. /255
)

#Entra al directorio de las imagenes del dataset-
training_image = train_datagen.flow_from_directory(
    data_training ,                                             #directorio del dataset.
    target_size = IMAGE_SIZE,                                   #procesa todas con sizes definidos.
    batch_size = batch_size,                                    #arriba definido como 32 (Default)
    class_mode= 'categorical'                                   #De forma categorica segun las etiquetas que tenemos.
)   

#Valida el validation
validation_image = validation_datagen.flow_from_directory(
    data_validation ,                                           #directorio del dataset.
    target_size = IMAGE_SIZE,                                   #procesa todas con sizes definidos.
    batch_size = batch_size,                                    #arriba definido como 32 (Default)
    class_mode= 'categorical'                                   #De forma categorica segun las etiquetas que tenemos.
)   

#Imprimimos los resultados de las imagenes
print(training_image.class_indices)

#Calculamos los steps_per_epoch
pasos_entrenamiento = training_image.n // training_image.batch_size
pasos_validation = validation_image.n // validation_image.batch_size

# ----------------------------------------- F I N ---------------------------------------------------------------

#-------------------------------------------TRAINING - NETWORK---------------------------------------------------
model = Sequential()#Modelo secuencial

#1 capa 
model.add(Convolution2D(32 , (3 , 3) , padding='same' ,  input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS) , activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#2 capa  
model.add(Convolution2D(64 , (3 , 3) , padding='same')) 
model.add(MaxPooling2D(pool_size=(2,2)))

#3 capa
model.add(Convolution2D(128 , (3 , 3) , padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

#4 capa pero densa
model.add(Flatten())
model.add(Dense(512 , activation='relu'))
model.add(Dropout(0.50))

#ultima capa
model.add(Dense(CLASSES , activation='softmax'))

#Entrenamiento de la red
model.compile(loss='categorical_crossentropy', 
            optimizer='rmsprop',
            metrics=['accuracy'])

tic=time.time()                                                     #para ver el tiempo que tomó el entrenamiento

H = model.fit(
    training_image, 
    steps_per_epoch = pasos_entrenamiento, 
    epochs = epoch, 
    validation_data  =validation_image, 
    validation_steps =pasos_validation)


print('Tiempo de procesamiento (secs): ', time.time()-tic)          #imprime el tiempo

history_dict = H.history
dictkeys=list(history_dict.keys())                                  #imprime las etiquetas [loss , accuracy, etc]
                          
#----------------------------------------------E N D -----------------------------------------------------------------


#Graficamos los resultados del entrenamiento -------------------------------------------------------------------------
#Graficamos los resultados del loss
N_epochs = epoch
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0 , N_epochs) , H.history["loss"] , label="Train_loss")
plt.plot(np.arange(0 , N_epochs) , H.history["val_loss"] , label="val_loss")
plt.title("Training - Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower right")
plt.savefig("Loss.png")

#Graficamos los resultados del accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0 , N_epochs) , H.history["accuracy"] , label="Train_acc")
plt.plot(np.arange(0 , N_epochs) , H.history["val_accuracy"] , label="val_acc")
plt.title("Training - Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.savefig("Accuracy.png")

#Graficamos los resultados del accuracy Y el loss
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0 , N_epochs) , H.history["accuracy"] , label="Train_acc")
plt.plot(np.arange(0 , N_epochs) , H.history["loss"] , label="Train_loss")
plt.title("Training - Accuracy and Loss")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy / Loss")
plt.legend(loc="center right")
plt.savefig("AccLoss.png")

#Graficamos los resultados del val_accuracy Y el val_loss
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0 , N_epochs) , H.history["val_accuracy"] , label="val_acc")
plt.plot(np.arange(0 , N_epochs) , H.history["val_loss"] , label="val_loss")
plt.title("Training - Val Accuracy and Val loss")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy / Loss")
plt.legend(loc="center right")
plt.savefig("AccLoss_val.png")


#----------------------------------------------------- E N D ---------------------------------------------------------



#------------------------------------------------- Ejecución del test --------------------------------------------------
print("\n\n\nExecution Test...")
test_datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

#generamos el pre procesamiento de las imagenes de test
test_generator = test_datagen.flow_from_directory(
    directory= "./DATASET/Test/",
    target_size = IMAGE_SIZE ,
    color_mode  = "rgb", 
    batch_size  = 1 ,
    class_mode= None, 
    shuffle= False ,
    seed= 42
)

#Calculamos el step_per_batch 
Step_size_test = test_generator.n // test_generator.batch_size
test_generator.reset()
pred = model.predict(test_generator , steps = Step_size_test , verbose = 1)                     #Predicción 

predicted_class_indices = np.argmax(pred , axis=1)

print(predicted_class_indices)
print(type(predicted_class_indices))

labels = (training_image.class_indices)
labels = dict((v , k)  for k , v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


filenames =  test_generator.filenames
results = pd.DataFrame({"Filename": filenames , 
                        "Predictions": predictions})
results.to_csv("results1.csv", index=False)
real_class_indices = []


for i in range(0 , len(filenames)):
    path = filenames[i]
    path_list = path.split(os.sep)

    if ("Banana" in path_list[1]):
        real_class_indices.append(0)
    if ("Manzana" in path_list[1]):
        real_class_indices.append(1)
    if ("Pera" in path_list[1]):
        real_class_indices.append(2)
    if ("Uva" in path_list[1]):
        real_class_indices.append(3)


real_class_indices = np.array(real_class_indices)
#--------------------------------------------------- E N D ----------------------------------------------------------


#--------------------- Graficamos los resultados con las imagenes del test ---------------------------------------------
fig = plt.figure(figsize=(30 , 30))
fig.subplots_adjust(hspace=0.1 , wspace=0.1)
rows = 10 
cols = len(filenames)//rows if len(filenames) % 2 == 0 else len(filenames)//rows +1
folder = "DATASET/Test/test_images/"

for i in range(0 , len(filenames)):
    your_path = filenames[i]
    path_lis = your_path.split(os.sep)
    img = mpimg.imread(folder + path_lis[1])
    ax = fig.add_subplot(cols , rows , i+1)
    ax.axis('Off')
    plt.imshow(img , interpolation=None)
    ax.set_title(predictions[i] , fontsize=20)
    plt.savefig("test.png")

#--------------------------------------------------- E N D ----------------------------------------------------------


#------------------------------------------Se guarda el entrenamiento----------------------------------------------------------------------------
dir = './Model/'
if not os.path.exists(dir):
    os.mkdir(dir)
model.save('./Model/model.h5')
model.save_weights('./Model/pesos.h5')
#--------------------------------------------------- E N D ----------------------------------------------------------


#--------------------------------------------------- Matriz de confusión --------------------------------------------
cm = confusion_matrix(real_class_indices, predicted_class_indices)
def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):
    
    plt.imshow(cm , interpolation='nearest' , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks , classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix , without normalized')
    
    print("CM: " , cm)

    thresh = cm.max() / 2.
    for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j , i , cm[i , j],
        horizontalalignment = "center",
        color = "white" if cm[i , j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

cm_plot_labels = training_image.class_indices
plot_confusion_matrix(cm , cm_plot_labels, title='Matriz de confusión')
#------------------------------------------------------ E N D --------------------------------------------------------
