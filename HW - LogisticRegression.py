# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 19:30:03 2022

@author: Adriana Suarez
"""

"""
email: adriana.suarezb@upb.edu.co
ID: 502197

"""
# Importamos las librerias
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# Datos
# ==============================================================================
# Llamamos los datos del dataset desde una url de github 
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/' \
       + 'Estadistica-machine-learning-python/master/data/spam.csv'
       
# Creamos el dataframe       
datos = pd.read_csv(url)

# # Mostramos los 3 primeros datos de la cabecera del dataframe
# datos.head(3)

# Modificamos la columna 'type' con 1 si es spam y 0 si no lo es, 

datos['type'] = np.where(datos['type'] == 'spam', 1, 0)

# Se identifica cuantas observaciones hay de cada clase.
print("Número de observaciones por clase")
print(datos['type'].value_counts())
print("")

# Se lee el porcentaje de las observaciones y se devuelve en orden de manera descendiente
print("Porcentaje de observaciones por clase")
print(100 * datos['type'].value_counts(normalize=True))


# División de los datos en train y test
# ==============================================================================
# Se ajusta un modelo de regresión logística múltiple con el objetivo 
# de predecir si un correo es spam en función de todas las variables disponibles.
X = datos.drop(columns = 'type')
y = datos['type']

# Dividimos los datos en dos bloques destinados al entrenamiento y validacion del modelo
X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
# Se hace la regresion logistica
modelo = sm.Logit(endog=y_train, exog=X_train,)
# Ajustamos los parametros de la regresion
modelo = modelo.fit()
# Imprimimos el modelo resumido
print(modelo.summary())

# Predicciones con intervalo de confianza 
# ==============================================================================
# Realizamos la prediccion con los datos de entrenamiento
predicciones = modelo.predict(exog = X_train)

# Clasificación predicha
# ==============================================================================
# Clasificamos los datos predecidos en 0.5 o 0.1
clasificacion = np.where(predicciones<0.5, 0, 1)
clasificacion

# Accuracy de test del modelo 
# ==============================================================================
#A la matriz de predictores se le tiene que añadir una columna de 1s para el test del modelo
X_test = sm.add_constant(X_test, prepend=True)
# Se realizan las prdicciones usando los datos de prueba
predicciones = modelo.predict(exog = X_test)
# Clasificamos/condicionamos los datos predecidos en 0.5 o 0.1
clasificacion = np.where(predicciones<0.5, 0, 1)
#  Medimos el porcentaje de predicciones correctas.
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = clasificacion,
            normalize = True
           )
print("")
# Se calcula el porcentaje de aciertos que tiene el modelo al predecir las observaciones de test (accuracy).
print(f"El accuracy de test es: {100*accuracy}%")

# Matriz de confusión de las predicciones de test
# ==============================================================================
# Devuelvemos la tabla de contingencia resultante en una matrix plana contigua
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    clasificacion,
    rownames=['Real'],
    colnames=['Predicción']
)
# Imprimimos la matrix
confusion_matrix