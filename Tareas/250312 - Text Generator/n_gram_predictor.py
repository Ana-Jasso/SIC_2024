#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Generación de Texto y Autocompletado con n-Gramas

Este script implementa un sistema básico de autocompletado y generación de texto utilizando n-gramas.
Los n-gramas son secuencias contiguas de n elementos (palabras en este caso) que aparecen en un texto.
Este enfoque es la base de muchos sistemas predictivos de texto.

Aplicaciones prácticas:
* Sistemas de autocompletado en teclados de smartphones
* Sugerencias de escritura en editores de texto
* Generación básica de texto
* Comprensión de los fundamentos de los modelos de lenguaje
"""

# Importamos las bibliotecas necesarias
import requests
from bs4 import BeautifulSoup
import nltk                                    # Biblioteca para procesamiento de lenguaje natural
import re
from numpy.random import randint, seed         # Funciones para generar números aleatorios
from sklearn.feature_extraction.text import CountVectorizer  # Para extraer n-gramas del texto

def get_text_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error al acceder a {url}")
        return ""
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.find('div', {'id': 'mw-content-text'})
    if not content:
        return ""
    
    for element in content(['script', 'style', 'table', 'sup']):
        element.decompose()
    
    text = ' '.join(p.get_text() for p in content.find_all('p')).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Eliminar puntuación
    return text

"""
## 1. Sistema de autocompletado basado en n-Gramas

### Fundamentos de los n-Gramas

Los n-gramas son secuencias contiguas de n elementos (palabras o caracteres) que se extraen de un texto. Por ejemplo:

* Unigramas (n=1): palabras individuales como "machine", "learning", "is"
* Bigramas (n=2): secuencias de dos palabras como "machine learning", "learning is"
* Trigramas (n=3): secuencias de tres palabras como "machine learning is"

El valor de los n-gramas radica en que capturan el contexto local. Esto permite predecir qué palabra 
podría seguir a una secuencia dada, basándose en patrones observados en datos de entrenamiento.
"""

# Texto de entrenamiento: Un párrafo sobre machine learning
# Este texto servirá como nuestros datos para extraer patrones de n-gramas
url = "https://es.wikipedia.org/wiki/Aprendizaje_autom%C3%A1tico"
my_text = get_text_from_url(url)

# Convertimos el texto a minúsculas y lo colocamos en una lista
# CountVectorizer espera una lista de documentos, por eso necesitamos este formato
my_text = [my_text.lower()]

"""
### 1.1. Extracción de n-Gramas

Primero, extraeremos todos los n-gramas del texto para entender su estructura. 
Usaremos la clase `CountVectorizer` de scikit-learn, que facilita la extracción de n-gramas.

Para este ejemplo, utilizaremos n=3 (trigramas), lo que significa que extraeremos 
secuencias de 3 palabras consecutivas.
"""

# Configuración para obtener trigramas (n=3)
n = 3                                              # Número de palabras en cada n-grama
n_min = n                                          # Tamaño mínimo de n-grama a extraer
n_max = n                                          # Tamaño máximo de n-grama a extraer
n_gram_type = 'word'                               # Tipo de n-grama: con palabras (también podría ser 'char')

# Creamos el vectorizador configurado para extraer trigramas
vectorizer = CountVectorizer(ngram_range=(n_min,n_max), analyzer=n_gram_type)

# Extraemos los n-gramas del texto
n_grams = vectorizer.fit(my_text).get_feature_names_out()  # Obtenemos los n-gramas como una lista
n_gram_cts = vectorizer.transform(my_text).toarray()       # La salida es un array de arrays
n_gram_cts = list(n_gram_cts[0])                           # Convertimos a una lista simple

# Mostramos los primeros 20 trigramas con sus frecuencias
print("Los primeros 20 trigramas encontrados en el texto:")
ngram_freq = list(zip(n_grams, n_gram_cts))
for i, (ngram, count) in enumerate(ngram_freq[:20]):
    print(f"{i+1}. '{ngram}' - aparece {count} veces")

"""
### 1.2. Creación del Diccionario de Predicción

Ahora, crearemos un diccionario que utilizaremos para predecir la siguiente palabra. Este diccionario tendrá la siguiente estructura:

* Clave: Un bigrama (las primeras dos palabras de un trigrama)
* Valor: Una lista de palabras que podrían seguir a ese bigrama (la tercera palabra de cada trigrama)

Este enfoque nos permite, dadas dos palabras consecutivas (un bigrama), predecir qué palabra podría 
venir a continuación basándonos en el texto de entrenamiento.
"""

# Creamos el diccionario para predecir la siguiente palabra
my_dict = {}

for a_gram in n_grams:                              # Para cada trigrama en nuestra lista
    words = a_gram.split()                          # Dividimos el trigrama en palabras individuales usando split()
    a_nm1_gram = ' '.join(words[0:n-1])             # Tomamos las primeras n-1 palabras (bigrama)
    next_word = words[-1]                           # Última palabra del trigrama (la que sigue al bigrama)
    
    if a_nm1_gram not in my_dict.keys():            # Si el bigrama no está en el diccionario
        my_dict[a_nm1_gram] = [next_word]           # Lo agregamos como nueva clave con su palabra siguiente
    else:                                           # Si el bigrama ya existe
        my_dict[a_nm1_gram] += [next_word]          # Añadimos la palabra siguiente a la lista existente

# Mostramos algunos ejemplos del diccionario para entender su estructura
print("\nEjemplos de bigramas y las palabras que pueden seguirlos:")
example_keys = list(my_dict.keys())[:10]  # Tomamos los primeros 10 bigramas
for key in example_keys:
    print(f"Bigrama: '{key}' → Posibles palabras siguientes: {my_dict[key]}")

# Verificamos bigramas con múltiples opciones para la siguiente palabra
print("\nBigramas con múltiples palabras posibles:")
for key, values in my_dict.items():
    if len(values) > 1:
        print(f"Bigrama: '{key}' → Opciones: {values}")

"""
### 1.3. Predicción de la Siguiente Palabra

Con nuestro diccionario preparado, ahora podemos crear una función que prediga la siguiente 
palabra dadas las dos palabras anteriores (un bigrama).

El proceso de predicción es:
1. Recibimos un bigrama como entrada
2. Buscamos en el diccionario todas las palabras que han seguido a ese bigrama en el texto
3. Seleccionamos aleatoriamente una palabra de esa lista

Esta selección aleatoria simula una selección probabilística, ya que las palabras más frecuentes 
tendrán más entradas en la lista y por tanto mayor probabilidad de ser seleccionadas.
"""

# Función para predecir la siguiente palabra dado un bigrama
def predict_next(a_nm1_gram):
    """
    Predice la siguiente palabra dado un bigrama (dos palabras).
    
    Args:
        a_nm1_gram (str): Un bigrama (dos palabras separadas por espacio)
        
    Returns:
        str: La palabra predicha que podría seguir al bigrama
    """
    # Obtenemos la longitud de la lista de palabras que pueden seguir al bigrama
    value_list_size = len(my_dict[a_nm1_gram])         
    
    # Generamos un índice aleatorio para seleccionar una palabra de la lista
    # El rango es de 0 a value_list_size - 1
    if value_list_size == 1:
        i_pick = 0  # Si solo hay una opción, seleccionamos esa
    else:
        i_pick = randint(0, value_list_size-1)  # Selección aleatoria entre las opciones disponibles
    
    # Devolvemos la palabra seleccionada
    return my_dict[a_nm1_gram][i_pick]

# Prueba con algunos bigramas específicos
test_bigrams = ['machine learning', 'order to', 'field of']
print("\nPruebas de predicción:")
for bigram in test_bigrams:
    if bigram in my_dict:
        print(f"Si veo '{bigram}', predigo que la siguiente palabra será: '{predict_next(bigram)}'")
    else:
        print(f"'{bigram}' no está en nuestro diccionario de bigramas")

# Demostración de la naturaleza probabilística
print("\nDemostración de predicciones múltiples para 'machine learning':")
bigram = 'machine learning'
predictions = {}
for i in range(20):
    next_word = predict_next(bigram)
    if next_word in predictions:
        predictions[next_word] += 1
    else:
        predictions[next_word] = 1

# Mostrar distribución de predicciones
for word, count in predictions.items():
    print(f"'{word}': {count} veces ({count/20*100:.1f}%)")

"""
### 1.4. Generación de Secuencias de Texto

Ahora usaremos nuestro modelo para generar una secuencia completa de texto automáticamente. El proceso es:

1. Comenzamos con un bigrama semilla (por ejemplo, "machine learning")
2. Utilizamos nuestro modelo para predecir la siguiente palabra
3. Añadimos esta palabra a nuestra secuencia
4. Tomamos las dos últimas palabras de nuestra secuencia como nuevo bigrama
5. Repetimos los pasos 2-4 hasta que encontremos un bigrama que no esté en nuestro diccionario

Este enfoque es similar a cómo funcionan algunos modelos de lenguaje más sofisticados, 
pero a una escala mucho más simple.
"""

# Inicializamos la semilla aleatoria para obtener resultados reproducibles
seed(123)

# Función para generar texto a partir de un bigrama semilla
def generate_text(seed_bigram, max_words=100):
    """
    Genera una secuencia de texto a partir de un bigrama semilla.
    
    Args:
        seed_bigram (str): El bigrama inicial para comenzar la generación
        max_words (int): Número máximo de palabras a generar
        
    Returns:
        str: El texto generado
    """
    if seed_bigram not in my_dict:
        return f"Error: '{seed_bigram}' no es un bigrama válido en nuestro diccionario."
    
    current_bigram = seed_bigram
    output_text = seed_bigram
    word_count = len(seed_bigram.split())
    
    # Bucle de generación
    while current_bigram in my_dict and word_count < max_words:
        # Predicimos la siguiente palabra
        next_word = predict_next(current_bigram)
        
        # Añadimos la palabra a nuestro texto
        output_text += " " + next_word
        word_count += 1
        
        # Actualizamos el bigrama para la siguiente iteración
        words = output_text.split()
        current_bigram = ' '.join(words[-2:])  # Tomamos las últimas dos palabras
    
    return output_text

# Generamos texto con diferentes bigramas semilla
seed_examples = ['machine learning', 'artificial intelligence', 'data mining']

for seed_bigram in seed_examples:
    if seed_bigram in my_dict:
        print(f"\n\nTexto generado a partir de '{seed_bigram}':\n")
        generated_text = generate_text(seed_bigram, max_words=50)
        print(generated_text)
    else:
        print(f"\n'{seed_bigram}' no es un bigrama válido en nuestro diccionario.")

"""
## Uso del Script

Para utilizar este script:

1. Asegúrate de tener las bibliotecas requeridas instaladas:
   pip install nltk scikit-learn numpy

2. Si deseas usar tu propio texto:
   - Modifica la variable `my_text` con tu texto de entrenamiento
   - Ajusta el valor de `n` según tus necesidades (2 para bigramas, 3 para trigramas, etc.)

3. Para generar texto con un bigrama semilla diferente:
   - Asegúrate de que el bigrama exista en tu texto
   - Llama a la función generate_text(tu_bigrama)

## Conclusiones y Posibles Mejoras

Este script implementa un modelo básico de lenguaje basado en n-gramas que puede usarse para:
- Predecir la siguiente palabra dado un contexto
- Generar texto automáticamente
- Comprender los fundamentos de los modelos de lenguaje estadísticos

Posibles mejoras:
1. Implementar selección ponderada por frecuencia en lugar de selección aleatoria
2. Añadir técnicas de suavizado para manejar n-gramas no vistos
3. Combinar n-gramas de diferentes tamaños para mejorar predicciones
4. Entrenar con corpus más grandes y diversos
"""

# Si se ejecuta como script principal
if __name__ == "__main__":
    print("\n\nEjemplo de ejecución completa:")
    
    # Solicitar al usuario un bigrama para comenzar
    print("\nIngresa un bigrama para generar texto (dos palabras separadas por espacio):")
    print("Ejemplos válidos: 'machine learning', 'data mining', 'it is'")
    
    try:
        user_input = input("> ").strip().lower()
        
        if user_input in my_dict:
            print(f"\nGenerando texto a partir de '{user_input}'...\n")
            result = generate_text(user_input, max_words=75)
            print(result)
        else:
            print(f"\nEl bigrama '{user_input}' no existe en el diccionario.")
            print("Bigramas disponibles (primeros 10):")
            for i, bigram in enumerate(list(my_dict.keys())[:10]):
                print(f"- {bigram}")
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        print(f"\nError: {e}") 