LABORATORIO 2 ESTRUCTURA DE DATOS

Jose Daniel Mestre Saucedo
Sergio Andres Pérez cacabelos
Juan David Arbelaez Merizalde



DESCRIPCION DEL LABORATORIO:

Debe construir un grafo simple, no dirigido y ponderado que modele las rutas de
transporte aéreo utilizando las coordenadas geográficas de varios aeropuertos.
Para ello, debe investigar sobre cómo obtener la distancia entre dos coordenadas
geográficas ya que este valor será el peso de la arista que conecte a dos
aeropuertos.
Los datos con los que debe construir el grafo se encuentran en el dataset
flights_final.csv que contiene la información relevante de los vuelos entre diversos
aeropuertos del mundo. Este dataset cuenta con 66930 registros y contiene los
siguientes atributos:
• Source Airport Code: Código del aeropuerto origen.
• Source Airport Name: Nombre del aeropuerto origen.
• Source Airport City: Ciudad del aeropuerto origen.
• Source Airport Country: País del aeropuerto origen.
• Source Airport Latitude: Latitud geográfica del aeropuerto origen.
• Source Airport Longitude: Longitud geográfica del aeropuerto origen.
• Destination Airport Code: Código del aeropuerto destino.
• Destination Airport Name: Nombre del aeropuerto destino.
• Destination Airport City: Ciudad del aeropuerto destino.
• Destination Airport Country: País del aeropuerto destino.
• Destination Airport Latitude: Latitud geográfica del aeropuerto destino.
• Destination Airport Longitude: Longitud geográfica del aeropuerto destino.
Utilizando las coordenadas geográficas de cada aeropuerto se debe mostrar un
mapa con la geolocalización de cada uno de ellos.
Sobre el grafo generado, el usuario debe poder realizar (desde la consola o desde
la interfaz gráfica) las siguientes acciones:
1. Determinar si el grafo generado es conexo. En caso de no serlo determinar el
numero de componentes y la cantidad de vértices de cada una de ellas.
2. Determinar el peso del árbol de expansión mínima. En caso de haber más de
una componente determinar el peso del árbol de expansión mínima de cada
una de las componentes.
3. Dado un primer vértice por el código del aeropuerto o seleccionado mediante
la interfaz gráfica:
a. Mostrar la información correspondiente al aeropuerto (código, nombre,
ciudad, país, latitud y longitud).
b. Mostrar la información (código, nombre, ciudad, país, latitud y longitud)
de los 10 aeropuertos cuyos caminos mínimos desde el vértice dado sean
los más largos. Adicionalmente, se debe mostrar la distancia de los
caminos.
4. Dado un segundo vértice por el código del aeropuerto o seleccionado
mediante la interfaz gráfica:
a. Mostrar el camino mínimo entre el primer y el segundo vértice sobre el
mapa de la interfaz gráfica, pasando por cada uno de los vértices
intermedios del camino. Para cada vértice intermedio se debe mostrar la
información del aeropuerto (código, nombre, ciudad, país, latitud y
longitud).
Para tener en cuenta:
• La solución (análisis) debe ser original.
• Puede utilizar Python, Java o el lenguaje de su elección para desarrollarlo.
• NO se puede utilizar librerías para realizar el cálculo de los caminos mínimos ni
del árbol de expansión mínima.
• Todos los códigos deben estar documentados por los integrantes del equipo.
• Este laboratorio es para desarrollar en grupos de 3 integrantes.
• Todo el código debe ser subido a un repositorio de GitHub.
• La sustentación del laboratorio se realizará durante la clase de laboratorio de la
semana de la entrega.
• Todos los integrantes del equipo deben estar presente
