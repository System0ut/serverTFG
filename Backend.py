from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os

model = load_model('modelo_setas.h5')

def cargar_y_preprocesar_imagen(ruta_imagen, tamaño_imagen=(224, 224)):
    # Cargar la imagen
    img = image.load_img(ruta_imagen, target_size=tamaño_imagen)
    # Convertir la imagen a un arreglo numpy
    img_array = image.img_to_array(img)
    # Agregar una dimensión extra para representar el lote (batch size)
    img_array = np.expand_dims(img_array, axis=0)
    # Escalar los píxeles al rango [0, 1]
    img_array /= 255.0
    return img_array

# Ruta de la imagen a predecir
# ruta_imagen = 'assets/fotoRandom2.jpg'
def predecirEspecie(rutaImagen):
    # Preprocesar la imagen
    imagen_preprocesada = cargar_y_preprocesar_imagen(rutaImagen)

    # Hacer la predicción
    umbral = 0.7
    prediccion = model.predict(imagen_preprocesada)

    # Obtener la clase con la mayor probabilidad
    probabilidad_max = np.max(prediccion)
    clase_predicha = np.argmax(prediccion, axis=1)

    # Obtener el nombre de las clases (en el orden en que aparecen en el generador)
    nombres_clases = ['Amanita muscaria: Venenosa','Fomes fomentarius: Venenosa','Fomitopsis pinicola: Toxica','Hypogymnia physodes: Comestible',
                      'Laetiporus sulphureus: Comestible','Parmelia sulcata: Venenosa','Xanthoria parietina: Medicinal']

    # Mostrar el nombre de la clase predicha
    if probabilidad_max < umbral:
        return 'Desconocida'
    else:
        nombre_clase_predicha = nombres_clases[clase_predicha[0]]
        return nombre_clase_predicha
