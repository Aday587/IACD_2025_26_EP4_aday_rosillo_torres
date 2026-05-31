Red Neuronal de Funciones de Base Radial - Aproximación de Funciones.

    -Descripción-

Este proyecto utiliza una Red Neuronal de Funciones de Base Radial para
aproximar la función f(x, y) = sin(sqrt(x^2 + y^2)).
Este trabajo se ha realizado para la asignatura Inteligencia Artificial para Ciencia de Datos.

    -Arquitectura-

El modelo sigue una estructura sencilla:

*   Datos
*   K-Means
*   Centroides
*   Funciones Gaussianas RBF
*   Matriz Z
*   ADALINE
*   Salida

    -Detalles de implementación-

*   Lenguaje: Python
*   Librerías: NumPy, Pandas, Matplotlib
*   Algoritmo de clustering: K-Means
*   Función de activación: Gaussiana
*   Capa de salida: ADALINE con regla de Widrow-Hoff

    -Experimentación-

Se ha estudiado el comportamiento del modelo variando el número de neuronas ocultas (K):

*   2
*   5
*   10
*   15
*   20

El rendimiento se evalúa mediante el error cuadrático medio.

    -Resultados-

El modelo mejora su capacidad de aproximación al aumentar el número de neuronas ocultas,
reduciendo el error hasta estabilizarse.

    -Visualización-

El proyecto incluye:

*   Representación en 2D del número de neuronas ocultas frente al error
*   Representación en 3D de la función real y la aproximación realizada por la red RBF

Ejecución del proyecto:

Para ejecutar el proyecto principal, utiliza el comando: python main.py
Para ejecutar el experimento, utiliza el comando: python Experiment.py
Para visualizar la representación 3D, utiliza el comando: python Plot3D.py

Trabajo práctico módulo 4.