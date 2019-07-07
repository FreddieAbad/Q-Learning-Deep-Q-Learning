<h1 align="center">  Q-Learning & Deep-Q-Learning </h1>
Uso de Algoritmo de Q-Learning &amp; Deep Q Learning para la resolucion del Juego Futbolin (Penales).

<h3 align="center"> Proyecto Final de Machine Learning </h3>
<h5 align="center"> Abad L. Freddy, Cabrera C. Edwin, Campoverde C. Daniel </h5>
<h5 align="center"> Facultad de Ingenieria, Escuela de Sistemas </h5>
<h5 align="center"> Universidad de Cuenca </h5>


##### Resumen.- 

Empleamos los algoritmos Q-Learning y Deep Q-Learning en la resolución del problema “La Final”, un pequeño juego donde un arquero de fútbol aprende, mediante aprendizaje reforzado, a atajar tiros de penal. Encontramos una configuración óptima de los parámetros influyentes del algoritmo Q-Learning para el problema particular, que logra un comportamiento libre de errores en aproximadamente 1500 episodios. Descubrimos que la solución empleando Deep Q-Learning describe una curva de aprendizaje similar, aunque incurre en un costo computacional y temporal mucho más alto a cambio de la generalización adicional que otorga.


#### Palabras Clave: 
Q-Learning, Aprendizaje, Reinforcement Learning, Deep.


## Introducción


Se pretende emplear el algoritmo Q-Learning y su variación Deep Q-Learning para solucionar el problema “La final”.
#### Definición del problema
“La Final” es un pequeño juego donde un arquero de fútbol aprende a atajar tiros de penal. Tanto el arquero como el balón se mueven en unidades espaciales discretas dentro de una rejilla de 9x8 recuadros. El problema se rige al siguiente conjunto de reglas:

* Cuando el arquero ataja, se ganan dos puntos (+2)

* Cuando el arquero no ataja, se pierden dos puntos (−2)

* Cuando la pelota es disparada fuera del arco y el arquero se mueve a ese lugar, se pierde un punto (−1). Tenga en cuenta que el arquero sí podría moverse, pero no fuera del arco (vea el punto 4)

* Cuando la pelota es disparada fuera del arco y el arquero no se sale del arco, se gana un punto (+1)

* Los disparos del balón son realizados enseguida uno después del otro. Esto quiere decir que el arquero, luego de un disparo, no se posiciona al centro del arco sino que trata de atajar el siguiente disparo desde el lugar en el que quedó luego del primer disparo.

## Marco Teórico
El aprendizaje reforzado es una área de machine learning inspirada en la psicología del comportamiento, en esta se modela el entorno como un proceso de decisión de Markov (MDP), donde ciertas acciones tomadas por el agente modifican el estado y suponen una cierta recompensa. El aprendizaje supervisado es particularmente apropiado para problemas que involucran beneficios a corto y largo plazo donde el beneficio total obtenido de ambos a de ser maximizado.

Q-Learning es una técnica de aprendizaje reforzado de tipo model-free, es decir, no requiere que el entorno sea previamente formulado en un modelo. Puede ser empleado para encontrar reglas de acción óptimas para un proceso de decisión de Markov aprendiendo una función Q(s,a) que representa el beneficio de tomar una acción a en un estado s.

El uso de técnicas de aprendizaje reforzado permite crear soluciones para tareas específicas, sin embargo, es altamente deseable encontrar soluciones generalizadas que sean aplicables en varias tareas diversas. Con este objetivo, Deep Q-Learning combina Q-Learning con aprendizaje profundo, empleando estas para representar la tabla Q. Para ello; se construye una red neuronal convolucional que puede ser aplicada en diferentes entornos. Un entorno se compone del actor (red neuronal en el entorno), un conjunto de movimientos posibles, y las penalizaciones y recompensas.

## Materiales y Métodos

Desarrollamos la solución al problema planteado usando el lenguaje python y la librería pygame para el control del aspecto gráfico del juego.

El juego emplea la división de cuadrículas para el desplazamiento de los elementos, según la especificación, la interfaz resultante se observa en la Figura 1.

![fig1](https://user-images.githubusercontent.com/38579765/60772861-f9b48380-a0c1-11e9-8001-458c25aeffb7.png)

Figura 1: Interfaz del Juego


La solución se estructura en 3 clases, según se observa en la Figura 2. El estado de las instancias correspondientes soporta el algoritmo de aprendizaje y el control gráfico del juego, proceso que se ejecuta en un bucle de episodios.




![fig2](https://user-images.githubusercontent.com/38579765/60772864-0e911700-a0c2-11e9-95d0-38331b1b4131.png)

Figura 2: Diagrama de clases

El funcionamiento general a alto nivel de la solución se aprecia en la Figura 2; Existe una matriz Q de dimensiones |S| x |A|, donde |S| es el número de estados posibles en el juego y |A| el número de acciones posibles. Cada elemento de la matriz Q representa el beneficio de tomar una acción a desde cierto estado s.

Para la instancia particular del juego planteado, un estado s se define como la combinación de las posiciones en el eje X del arquero y del balón, y una acción a como el movimiento del arquero por  pasos = n  {-9 ... 9}, donde un factor n negativo supone un movimiento de n pasos a la izquierda, un factor positivo supone n pasos a la derecha y un factor de n = 0 supone no moverse.

La matriz de aprendizaje Q se implementa como un diccionario que asocia instancias de Estado a vectores de acciones que representan el movimiento del Arquero sobre el eje X.

En cada episodio del juego, esto es, en cada tiro de penal, se identifica el estado actual e, luego se toma de la matriz Q la mejor acción posible para dicho estado, la misma que permite saltar a un nuevo estado definido e’.

Tras patear el balón, se aplican el conjunto de reglas previamente definidas para evaluar el reward (recompensa), producto de la acción del arquero, con el cual se actualiza la matriz de aprendizaje Q según la Ecuación 1, donde lr representa el learning rate y gamma el factor de descuento para el beneficio de estados futuros.

![form1](https://user-images.githubusercontent.com/38579765/60772888-48621d80-a0c2-11e9-9e7d-3166457e8de8.png)


El parámetro lr influye en la sensibilidad del algoritmo a recibir nuevo conocimiento, esto es, en qué medida el algoritmo altera los patrones de comportamiento aprendidos con cada refuerzo recibido. El parámetro gamma altera la sensibilidad del algoritmo a preferir beneficios futuros por sobre beneficios inmediatos o viceversa, que ocurren como resultado de su decisión de una acción inmediata y cómo esta altera su elección de acciones futuras.



![fig3](https://user-images.githubusercontent.com/38579765/60772867-194bac00-a0c2-11e9-9782-0e1eae0f5151.png)

Figura 3: Funcionamiento de alto nivel



##  Optimización de parámetros influyentes

Los parámetros learning rate (lr) y Gamma, definidos para x  {0, 1}  x  , configuran y alteran el comportamiento del proceso de aprendizaje. Con propósitos de optimización, probamos multiples configuraciones y comparamos el Puntaje obtenido como resultado de aplicar las reglas planteadas contra un número de Episodios fijo de 2000. Según se observa en la Figura 4, el puntaje más alto para el número definido de Episodios es de 26306  y ocurre para un Learning rate = 1  y  Gamma  = 0.

![fig4](https://user-images.githubusercontent.com/38579765/60772870-25376e00-a0c2-11e9-86bd-3fc827d9dddc.png)

Figura 4: Optimización de parámetros

El uso de estos parámetros, dan como resultado una curva de aprendizaje que relaciona el número de episodios (tiros de penal) con el puntaje acumulado obtenido, según se observa en la Figura 5.

![fig5](https://user-images.githubusercontent.com/38579765/60772877-308a9980-a0c2-11e9-8e26-a44095cd9dd6.png)

Figura 5: Curva de aprendizaje


## Deep Q-Learning

La solución empleando el algoritmo Deep Q-Learning comparte los mismos principios empleados en la solución con el algoritmo Q-Learning e incrementa consideraciones específicas a este:

El número de frames capturados por segundos representan un estado del entorno en formato de imagen, es por ello que se la recopilación de dichas imágenes representan una mayor carga computacional.

Al implementarse una red neuronal, el algoritmo tiende a “olvidar” debido al ajuste de los pesos de las neuronas en las capas intermedias de la red, es por ello; que se requiere almacenar los estados anteriores.


La naturaleza de la solución supone un reconocimiento del estado del juego a partir de capturas individuales de la imagen generada, por tal motivo la interfaz propuesta (Figura 6) para esta versión de la solución es mucho más simple, lo que facilita el trabajo de la red neuronal convolucional involucrada

![fig6](https://user-images.githubusercontent.com/38579765/60772883-3c765b80-a0c2-11e9-8437-85285566091e.png)

Figura 6: Interfaz del Juego (DQL)

## Resultados y Discusión

El puntaje máximo obtenido para la configuración explorada en la Figura 4 (Learning rate = 1  y  Gamma  = 0), concuerda con lo que cabría esperar dada la naturaleza del problema: El aprendizaje no obtiene ningún beneficio al analizar el impacto de acciones futuras en nuevos tiros de penal, sino únicamente del beneficio de la acción inmediata, situación que se maximiza con un Gamma = 0. Por otro lado cada refuerzo, tanto negativo como positivo, lleva consigo información de aprendizaje que ha de ser aprovechada de forma inmediata en lugar de retrasarla, situación que se maximiza con un lr = 1.

La curva de aprendizaje de la Figura 5, indica un descenso del puntaje acumulado durante los primeros ~300 episodios, mientras el arquero toma decisiones al azar; El mismo es también un punto de inflexión a partir del cual el arquero adquiere un comportamiento que le permite incrementar su puntaje, hasta que en el episodio ~1500  la relación entre el puntaje acumulado y los tiros de penal se torna linealmente creciente, esto es, el arquero realiza siempre la acción óptima.

La solución empleando  el algoritmo Deep Q-Learning describe una curva de aprendizaje similar a la observada para la solución con Q-Learning. Esta ocurre, sin embargo, no pudo ser observada en su totalidad, dado el costo computacional implicado y la falta de recursos durante la experimentación. 

## Conclusiones

Se emplearon los algoritmos Q-Learning y Deep Q-Learning para la resolución del problema planteado, y se estudiaron las configuraciones de sus parámetros influyentes, permitiéndonos obtener el máximo aprendizaje posible por episodio. Descubrimos que la configuración óptima para el problema resuelto ocurre con los parámetros lr = 1; Gamma = 0, la misma que logra un comportamiento libre de errores en aproximadamente 1500 episodios.

Dado el comportamiento aprendido para el número de episodios de entrenamiento requerido, consideramos que los algoritmos empleados ofrecen un excelente rendimiento y aplicabilidad al problema resuelto. 

##  Bibliografía

* Keon. 2017. Deep Q-Learning with Keras and Gym

* Matiisen. 2015. Demystifying Deep Reinforcement Learning

* Mnih. et. al.  2013. Playing atari with deep reinforcement Learning

* Salter D. 2016. Deep Q-Learning Pong with Python & TensorFlow.
