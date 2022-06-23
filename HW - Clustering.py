# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 10:25:51 2022

email: adriana.suarezb@upb.edu.co
ID: 502197


@author: Adriana Suarez
"""

# Tarea clustering 
# Importamos las librerias
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Se crea la funcion para tomar la informacion de AgglomerativeClustering
def plot_dendrogram(model, **kwargs):
    '''
    Esta función extrae la información de un modelo AgglomerativeClustering
    y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
    '''
    # Se crea una variable que guarde/devuelva un array con ceros 
    counts = np.zeros(model.children_.shape[0])
    # Se lee la longitud del modelo con los labels
    n_samples = len(model.labels_)
    # Se crea un ciclo con una funcion que retorne dos valores para cada iteracion
    for i, merge in enumerate(model.children_):
        # Se inicializa una variable contador
        current_count = 0
        # Se crea un for dentro del anterior que recorra el merge 
        for child_idx in merge:
            #Ahora se crea un condicional de child_idx, donde si esta variable es menor
            # que n_samples se incrementara el contador
            if child_idx < n_samples:
                # Por cada concicion cumplida se incrementa el contador en 1
                current_count += 1  # leaf node
                # Sino, el contador restara los chil_idx de los n_samples dentro del primer contador
                # que contenia el array de ceros
            else:
                current_count += counts[child_idx - n_samples]
                # La variable que guarda el array de ceros iterara en i y retornara el valor del contador
        counts[i] = current_count

# Se crea una variable que apilara las matrices 1-D como columnas en una matrz 2-D
# y vinculara las variables del modelo children
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot
    #Se crea el dendrograma
    dendrogram(linkage_matrix, **kwargs)
    
    
# Simulación de datos
# ==============================================================================
# Dentro de las variables X, y utilizaremos el make_blobs que nos generara los
# clusters de los datos con distribucion gausiana e isotropica
X, y = make_blobs(
        n_samples    = 200, 
        n_features   = 2, 
        centers      = 4, 
        cluster_std  = 0.60, 
        shuffle      = True, 
        random_state = 0
       )
# Se crea el metodo que devuelve un objeto de la figura y un objeto de los ejes, que se usan para dibujar o
# manipular el grafico
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
# Se crea un ciclo que recupere los valores unicos de el array en la variable y
for i in np.unique(y):
    #Dibujamos el diagrama de dispersion con los parametros guardados en xy
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        # Le damos nombre y color a los ejes y puntos de nuestra figura
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
    # Se le da un titulo a nuestra grafica
ax.set_title('Datos simulados')
# Se muestra la layenda de la grafica
ax.legend();


# Escalado de datos
# ==============================================================================
# Se realiza el escalado de los datos
X_scaled = scale(X)

# Modelos
# ==============================================================================
# Se crea un modelo aplicando un hierarchical clustering aglomerativo  se utiliza 
# el tipo de linkage complete
modelo_hclust_complete = AgglomerativeClustering(
    #Dentro de la metrica se utliza la distancia euclidean
                            affinity = 'euclidean',
                            # Se utiliza el tipo de linkage complete
                            linkage  = 'complete',
                            distance_threshold = 0,
                            #Determinamos el numero de cluster a usar, como el 
                            # distance_threshold es 0 su valor sera none
                            n_clusters         = None
                        )
# Ajustamos los parametros para hacer la regresion lineal a los datos con la escala
# dentro del modelo complete
modelo_hclust_complete.fit(X=X_scaled)

# Se crea un modelo aplicando un hierarchical clustering aglomerativo  se utiliza 
# el tipo de linkage average
modelo_hclust_average = AgglomerativeClustering(
    #Dentro de la metrica se utliza la distancia euclidean
                            affinity = 'euclidean',
                            # Se utiliza el tipo de linkage average
                            linkage  = 'average',
                            distance_threshold = 0,
                            #Determinamos el numero de cluster a usar, como el 
                            # distance_threshold es 0 su valor sera none
                            n_clusters         = None
                        )
# Ajustamos los parametros para hacer la regresion lineal a los datos con la escala
# dentro del modelo average
modelo_hclust_average.fit(X=X_scaled)

# Se crea un modelo aplicando un hierarchical clustering aglomerativo  se utiliza 
# el tipo de linkage ward
modelo_hclust_ward = AgglomerativeClustering(
    #Dentro de la metrica se utliza la distancia euclidean
                            affinity = 'euclidean',
                            # Se utiliza el tipo de linkage ward
                            linkage  = 'ward',
                            distance_threshold = 0,
                            #Determinamos el numero de cluster a usar, como el 
                            # distance_threshold es 0 su valor sera none
                            n_clusters         = None
                     )
# Ajustamos los parametros para hacer la regresion lineal a los datos con la escala
# dentro del modelo ward
modelo_hclust_ward.fit(X=X_scaled)

# Dendrogramas
# ==============================================================================
# Creamos los dendogramas
# Se crea el metodo que devuelve un objeto de la figura, un objeto de los ejes y tamaño,
# que se usan para dibujar o manipular el grafico
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
# Primero se crea el dendograma para el modelo average
plot_dendrogram(modelo_hclust_average, color_threshold=0, ax=axs[0])
axs[0].set_title("Distancia euclídea, Linkage average")
# Se crea el dendograma del modelo complete
plot_dendrogram(modelo_hclust_complete, color_threshold=0, ax=axs[1])
axs[1].set_title("Distancia euclídea, Linkage complete")
# Se crea el dendograma del modelo ward
plot_dendrogram(modelo_hclust_ward, color_threshold=0, ax=axs[2])
axs[2].set_title("Distancia euclídea, Linkage ward")
plt.tight_layout();

# Se crea el metodo que devuelve un objeto de la figura, un objeto de los ejes y tamaño,
# que se usan para dibujar o manipular el grafico
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# Se le da un valor al corte de la figura del dendograma de linkage ward
altura_corte = 6
# Se crea el dendograma del modelo ward
plot_dendrogram(modelo_hclust_ward, color_threshold=altura_corte, ax=ax)
# Se le pone titulo a la figura
ax.set_title("Distancia euclídea, Linkage ward")
# Se configuran los colores o labels
ax.axhline(y=altura_corte, c = 'black', linestyle='--', label='altura corte')
ax.legend();


# Método silhouette para identificar el número óptimo de clusters
# ==============================================================================
# Primero determinamos un range con una secuencia de numeros
range_n_clusters = range(2, 15)
# Creamos una lista vacia
valores_medios_silhouette = []
# Se crea un bucle para el numero de clusters a usar en el range definido anteriormente
for n_clusters in range_n_clusters:
    # Se crea un modelo aplicando un hierarchical clustering aglomerativo  se utiliza 
    # el tipo de linkage ward
    modelo = AgglomerativeClustering(
        #Dentro de la metrica se utliza la distancia euclidean
                    affinity   = 'euclidean',
                    linkage    = 'ward',
                    #Determinamos el numero de cluster a usar, 
                    n_clusters = n_clusters
             )
    
    # Se le da color a los clusters con el escalado
    cluster_labels = modelo.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    valores_medios_silhouette.append(silhouette_avg)
# Manipulamos la figura dandoles valores al objeto y determinando su tamaño    
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.plot(range_n_clusters, valores_medios_silhouette, marker='o')
# se le da titulo a la grafica y a los ejes de la figura
ax.set_title("Evolución de media de los índices silhouette")
ax.set_xlabel('Número clusters')
ax.set_ylabel('Media índices silhouette');





