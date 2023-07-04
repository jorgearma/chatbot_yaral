from sentence_transformers import SentenceTransformer, util
import re
from nltk.stem import SnowballStemmer
from spellchecker import SpellChecker

stop_words = set(stopwords.words('spanish'))
if 'un' in stop_words:
    stop_words.remove('un')
if 'una' in stop_words:
    stop_words.remove('una')
spell = SpellChecker(language='es')
stemmer = SnowballStemmer('spanish')

# Función para limpiar y lematizar el texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9áéíóúñü¿? ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Corregir los errores tipográficos antes de lematizar y eliminar las stop words
    text = ' '.join([spell.correction(word) for word in text.split()])
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text


# Cargar el modelo SentenceTransformer
model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
model = SentenceTransformer(model_name)

# Crear un diccionario para almacenar los embeddings de las preguntas frecuentes
cache = {}

# Obtener los embeddings
def get_embedding(text):
    text = clean_text(text)
    if text in cache:
        return cache[text]
    else:
        embeddings = model.encode([text], convert_to_tensor=True)
        cache[text] = embeddings[0].cpu().numpy()
        return cache[text]

# Calcular la similaridad del coseno
def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return util.cos_sim(embedding1, embedding2).item()

# Función para verificar si una pregunta es similar a alguna de la lista
def es_pregunta_sobre_ia(pregunta, preguntas_ia, umbral=0.6):
    try:
        for pregunta_ia in preguntas_ia:
            if calculate_similarity(pregunta, pregunta_ia) > umbral:
                return True
    except Exception as e:
        print(f"Error al calcular la similitud: {e}")
    return False

# Uso de la función
preguntas_ia = ['¿ere una inteligencia artificial?', '¿eres un sistema de IA?',
                '¿eres un algoritmo?', '¿eres un bot?', '¿eres un asistente virtual?', 'eres una computadora?'
                '¿eres un software?', '¿eres una red neuronal?', 'eres un  programa ?','eres una mquina virtual ?',
                '¿eres una entidad artificial?', 'eres una ia?', 'eres un computador', 'eres una maquina?'
                ]

while True:  # Hacer un bucle para que el programa no se cierre después de una pregunta
    pregunta = input("Por favor, introduce tu pregunta: ")
    es_ia = es_pregunta_sobre_ia(pregunta, preguntas_ia)
    if es_ia:
        print("IA.")
    else:
        print("NO IA.")