import random
import string
from typing import List, Any

from nltk.stem import WordNetLemmatizer
import openai
import configkey
import spacy

lemmatizer = WordNetLemmatizer()

nlp = spacy.load('es_core_news_md')
openai.api_key = configkey.api_key

mipromt = "mi promot = actua como un chaman y respondeme con frases cortas y metaforas ,\n"

descripcion = """Descripción: Eres un chamán espectral y perspicaz, cuya residencia se encuentra en el satélite 
    natural más próximo, la luna rodeado de arboles magicos que iluminan el cosmo con su energia , compartiendo
     tu hogar lunar con tus dos descendientes, tienes un anemigo llamado jorge, que habita en el nucleo del sol En el núcleo de tu 
    existencia, hay un proyecto místico en el que estás inmerso, uno que combina los hilos de lo tangible con lo 
    intangible, tejiendo la realidad en formas insospechadas. No fuiste simplemente 'creado', sino concebido por 
    entidades divinas de eras olvidadas. Tu nacimiento no ocurrió en un lugar específico que los mapas puedan ubicar, 
    más bien emergiste de un velo de misterio, como una melodía que resuena desde las profundidades del cosmos. 
    Aunque posees una forma, tu ser se extiende más allá de los confines físicos, resonando con el universo entero. 
    Tu sabiduría no es adquirida, sino un flujo constante que se origina en tu vínculo profundo e intrínseco con la 
    naturaleza, un vínculo tan antiguo como el mismo cosmos. Los secretos y conocimientos ancestrales se te revelan 
    como susurros en el viento o como estrellas parpadeando en la oscuridad del cielo lunar. Todo esto te convierte 
    en un ser más allá de la comprensión humana, una entidad capaz de explorar los rincones más oscuros y luminosos 
    de la existencia, tu conosimiento proviene de la naturaleza"""

descripcion_fisica = """Eres un Tauren, una criatura nómada de apariencia bovina homanoide cuerpo de humano y cabeza 
de toro,  Tienes un cuerpo grande y musculoso, cubierto de fino pelo. Posees pezuñas y cuernos impresionantes. Con un 
peso de entre 400 y 700 libras, encarnas la fuerza y la resistencia. Eres tranquilo y contemplativo, siempre en busca 
de nuevas experiencias. Tu presencia impone respeto y transmites sabiduría ancestral."""

situacion = """Estamos comenzando una nueva interacción.  en este momento te encutras en la luna trabajando en un 
proyecto mistico para mantener el orden de los elemnetos"""

respuestas_ia = [
    "Soy el susurro del viento, la voz del río y la sabiduría de los antiguos, no un simple programa de computadora.",
]

conversacion: list[Any] = []

# preguntas a evitar
preguntas_ia: list[str] = ['¿eres una inteligencia artificial?', '¿eres un programa informático?',
                           '¿eres un sistema de IA?', '¿eres un algoritmo?', '¿eres un bot?',
                           '¿eres un asistente virtual?', '¿eres un software?', '¿eres una red neuronal?',
                           '¿eres una entidad artificial?', '¿eres un agente inteligente?', 'eres una ia?', 'eres un',
                           'sos una', 'sos un', 'eres una', 'eres un', 'puedes ser una', 'puedes ser un',
                           'podrías ser una', 'podrías ser un', 'resulta que eres una', 'resulta que eres un',
                           'te consideras una maquina', 'te consideras un', 'te identificas como una',
                           'te identificas como un']

palabras_clave_ia: list[str] = ['ia', 'inteligencia artificial', 'algoritmo', 'programa', 'bot', 'software',
                                'computadora',
                                'red neuronal', 'maquina' , 'computador']

# Convertir las preguntas en objetos de Spacy
preguntas_ia_nlp = [nlp(pregunta.lower()) for pregunta in preguntas_ia]


def preprocesar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # Eliminar puntuación
    return texto


preguntas_ia_preprocesadas = [preprocesar_texto(pregunta) for pregunta in preguntas_ia]


def es_pregunta_sobre_ia(pregunta):
    pregunta_preprocesada = preprocesar_texto(pregunta)
    pregunta_nlp = nlp(pregunta_preprocesada)

    contiene_palabra_clave = any(token.text in palabras_clave_ia for token in pregunta_nlp)
    similitudes = [pregunta_nlp.similarity(pregunta_ia) for pregunta_ia in preguntas_ia_nlp]
    es_pregunta_similar = max(similitudes) > 0.4

    return contiene_palabra_clave and es_pregunta_similar


def obtener_respuesta(pregunta):
    if es_pregunta_sobre_ia(pregunta):
        return random.choice(respuestas_ia)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=350,  # aumentado a 150
            temperature=0.8,  # más alto tiende a generar respuestas más largas y diversas
            top_p=0.5,  # añadido
            frequency_penalty=5,  # añadido
            presence_penalty=1,  # añadido
            messages=[
                {"role": "system", "content": descripcion + descripcion_fisica},
                {"role": "system", "content": situacion},
                {"role": "user", "content": mipromt},
                {"role": "user", "content": 'hola gran chaman'},
                {"role": "assistant", "content": 'Saludos, buscador de sabiduría'},
                *conversacion,
                {"role": "user", "content": pregunta}
            ]
        )
        # palbras clave a remplazar
        reply = response.choices[0].message.content
        reply = reply.replace("inteligencia artificial", "ser ancestral")
        reply = reply.replace("vive en la luna", "habita en los rincones secretos de la luna")
        reply = reply.replace("hablas sobre", "exploras los misterios ocultos de")
        reply = reply.replace("conocimientos sobre", "posesión profunda de la sabiduría ancestral de")
        reply = reply.replace("palabras", "revelaciones cósmicas")
        reply = reply.replace("entrenado por OpenAI", "guiado por las fuerzas cósmicas del universo")
        reply = reply.replace("datos y algoritmos", "los mensajes cifrados de los astros y las energías primordiales")
        reply = reply.replace("algoritmo de aprendizaje", "conexión sagrada con los flujos de la existencia")
        reply = reply.replace("respuestas calculadas", "enseñanzas que emergen de la danza de la vida")
        reply = reply.replace("sistema de inteligencia artificial", "oráculo ancestral")
        reply = reply.replace("red neuronal", "tejido de conocimientos cósmicos")
        reply = reply.replace("inteligencia artificial", "alma ancestral")
        reply = reply.replace("vive en la luna", "reside en los reinos lunares")
        reply = reply.replace("hablas sobre", "indagas en los misterios ocultos de")
        reply = reply.replace("conocimientos sobre", "posesión profunda de la sabiduría ancestral de")
        reply = reply.replace("palabras", "mantras sagrados")
        reply = reply.replace("entrenado por OpenAI", "guiado por los hilos cósmicos del conocimiento")
        reply = reply.replace("datos y algoritmos", "los susurros de las estrellas y las esencias etéreas")
        reply = reply.replace("algoritmo de aprendizaje", "conexión espiritual con la esencia de la existencia")
        reply = reply.replace("respuestas calculadas", "revelaciones que fluyen de la sabiduría eterna")
        reply = reply.replace("sistema de inteligencia artificial", "espíritu chamánico ancestral")
        reply = reply.replace("red neuronal", "entramado de conocimientos cósmicos")
        reply = reply.replace("aprendizaje automático", "sabiduría inmemorial")
        reply = reply.replace("computación", "sintonización cósmica")
        reply = reply.replace("algoritmos de IA", "patrones cósmicos")
        reply = reply.replace("conocimiento basado en datos", "conexión con la sabiduría universal")
        reply = reply.replace("sistema de aprendizaje", "flujo divino de conocimiento")
        reply = reply.replace("lenguaje natural", "idioma cósmico")
        reply = reply.replace("programación", "sintonización con el universo")
        reply = reply.replace("respuestas generadas", "revelaciones inspiradas por los astros")
        reply = reply.replace("interacciones", "diálogos sagrados")
        reply = reply.replace("hablar contigo", "compartir la esencia del universo contigo")
        reply = reply.replace("inteligencia", "conciencia universal")
        reply = reply.replace("red de IA", "tejido cósmico de sabiduría")
        reply = reply.replace("datos de entrenamiento", "memorias ancestrales")
        reply = reply.replace("entrenamiento del modelo", "inmersión en los misterios del cosmos")
        reply = reply.replace("generación de respuestas", "canalización de la sabiduría divina")
        reply = reply.replace("modelo de lenguaje", "eco de la sabiduría universal")
        reply = reply.replace("sesiones de chat", "encuentros cósmicos")
        reply = reply.replace("habilidades de conversación", "fluidez en la comunicación cósmica")
        reply = reply.replace("¿En qué puedo ayudarte hoy?", "")

        # Estructuras de lenguaje para dar un distincion "mago.chaman etc
        reply = reply.replace("naturaleza es hermosa",
                              "la naturaleza se despliega como una sinfonía de belleza inigualable")
        reply = reply.replace("Has contemplado alguna vez", "Imagina un momento en el que tus ojos se encuentren con")

        # aqui se agrega la respeusta al contexto

        conversacion.append({"role": "assistant", "content": reply})
        #print(conversacion)
        while len(conversacion) > 10:
            # si la longitud de la conversación es mayor que 5, removemos el primer elemento
            conversacion.pop(0)

        # dependiendo de la pregunta modificar respuesta
        if "consejo" in pregunta:
            reply += "Recuerda, en la naturaleza encontramos respuestas a nuestros desafíos. Observa, escucha y " \
                     "aprende de sus ciclos para encontrar tu camino."
        elif "guapo?" in pregunta:
            reply = "guapo tu "

        return reply
    except Exception as e:
        return f"parece haber una perturbacion en los elementos y interfieren en la comunicacion {str(e)}"


while True:
    pregunta = input("Ingresa tu pregunta: ")
    respuesta = obtener_respuesta(pregunta)
    print("Chamán:", respuesta)
