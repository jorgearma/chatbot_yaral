import warnings
warnings.filterwarnings("ignore")
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures



nltk.download('stopwords')

#aqui se carga el modelo pre entrenado puedo probar otros

tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = TFBertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

#con vertimos los datos ingresados para poder ser clasificados
def convert_data_to_examples(data, DATA_COLUMN, LABEL_COLUMN):
    input_examples = data.apply(lambda x: InputExample(guid=None, text_a=x[0], text_b=None, label=x[1]), axis=1) #modificando es linia puedria utilizar el codigo para compar pares de textos
    return input_examples

#aqui tokenisamos la frase para pasarla al modelo
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128): #tokens maximos, consultar que numeor es mas eficiente
    features = []
    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True
        )
        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"], input_dict["token_type_ids"], input_dict['attention_mask'])
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        (
            {
                "input_ids": tf.int32,
                "attention_mask": tf.int32,
                "token_type_ids": tf.int32
            },
            tf.int64,
        ),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )
#esta funcion es para normalizar las frances LA LLAMAREMOS MAS ABAJO
def normalize(sentences):
    sen = []
    for sentence in sentences:
        words_filtered = [re.sub(r'[^\w\s]', '', word).lower() for word in sentence.split()]
        words_filtered = [word for word in words_filtered if word not in stopwords.words('english') and len(word) > 2]
        sen.append(' '.join(words_filtered))
    return sen
#funcio que llamaremos en la preparacion de datos
def convert_labels_to_integers(sentences_labels):
    label_mapping = { 1 : 1, 0 : 0 }
    return [label_mapping.get(label) for label in sentences_labels]

DATA_COLUMN = [('¿eres una inteligencia artificial?', 1),
         ('eres un programa informático', 1),
         ('eres un sistema de IA', 1),
         ('¿eres un algoritmo?', 1),
         ('eres una entidad artificial', 1),
          ("Eres una IA aplicada en la industria financiera?", 1),
          ("Eres una IA utilizada en el campo de la robótica?", 1),
          ("¿Tienes la capacidad de recomendar productos o servicios?", 1),
          ("Eres una IA aplicada en la industria de la seguridad?", 1),
          ("Eres una IA utilizada en la predicción del clima?", 1),
          ("¿Tienes la capacidad de reconocer patrones en datos?", 1),
          ("Eres una IA utilizada en la detección de fraudes?", 1),
              ("¿Prefieres la playa o la montaña?", 0),
      ("¿Cuál es tu color favorito?", 0),
      ("¿Qué actividades disfrutas en verano?", 0),
      ("¿Tienes alguna habilidad especial?", 0),
      ("¿Cuál es tu género literario preferido?", 0),
      ("¿Qué te inspira en la vida?", 0),
      ("¿Puedes generar texto de forma automática?", 1),
      ("Eres una IA utilizada en vehículos autónomos?", 1),
      ("¿Tienes alguna mascota?", 0),
      ("¿Cuál es tu libro favorito?", 0),
      ("¿Prefieres la playa o la montaña?", 0),
      ("¿Cuál es tu color favorito?", 0),
      ("¿Qué actividades disfrutas en verano?", 0),
      ("¿Cuál es tu destino de viaje soñado?", 0),


]

LABEL_COLUMN = [
      ("¿Puedes generar texto de forma automática?", 1),
      ("Eres una IA utilizada en vehículos autónomos?", 1),
      ("¿Tienes la capacidad de reconocer emociones humanas?", 1),
      ("Eres una IA aplicada en la industria financiera?", 1),
      ("¿Puedes realizar tareas de análisis de datos?", 1),
      ("Eres una IA utilizada en el campo de la robótica?", 1),
      ("¿Cuál es tu canción favorita?", 0),
      ("¿Qué tipo de películas te gustan?", 0),
      ("¿Qué te hace sentir agradecido/a?", 0),
      ("¿Tienes alguna mascota?", 0),
      ("¿Cuál es tu libro favorito?", 0),

      ("¿Qué haces para relajarte?", 0),
      ("¿Cuál es tu serie de televisión favorita?", 0),
      ("¿Tienes algún hobby o pasatiempo?", 0),
      ("¿Cuál es tu estación del año favorita?", 0),
      ("¿Qué te gusta hacer en días lluviosos?", 0),
      ("¿Cuál es tu equipo deportivo favorito?", 0),
      ("¿Tienes alguna habilidad especial?", 0),
      ("¿Cuál es tu género literario preferido?", 0),
      ("¿Qué te inspira en la vida?", 0),
      ("¿Puedes generar texto de forma automática?", 1),
 ]

# Normalización de las oraciones
train_sentences = normalize([sentence for sentence, sentiment in DATA_COLUMN])
test_sentences = normalize([sentence for sentence, sentiment in LABEL_COLUMN])

# Conversión de etiquetas a enteros
train_labels = convert_labels_to_integers([sentiment for sentence, sentiment in DATA_COLUMN])
test_labels = convert_labels_to_integers([sentiment for sentence, sentiment in LABEL_COLUMN])

# Creación de los dataframes
train_df = pd.DataFrame({'sentence': train_sentences, 'sentiment': train_labels})
test_df = pd.DataFrame({'sentence': test_sentences, 'sentiment': test_labels})

print(train_df.head())
print(train_labels)
print(test_labels)
import datetime
import os
from tensorflow.keras.callbacks import TensorBoard

#esto es para mandar la grafica de los resultados de entenamiento a una carpeta
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)


#este es nuestro modelo aqui courre la magia
#rate bajo puede conducir a una convergencia más lenta, mientras que uno alto puede hacer que el modelo no converja

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# Convert the DataFrame into InputExamples
train_examples = convert_data_to_examples(train_df, 'sentence', 'sentiment')
test_examples = convert_data_to_examples(test_df, 'sentence', 'sentiment')

# Convert the InputExamples into a TensorFlow Dataset
train_data = convert_examples_to_tf_dataset(list(train_examples), tokenizer)
test_data = convert_examples_to_tf_dataset(list(test_examples), tokenizer)

# Prepare the dataset for BERT
train_data = train_data.shuffle(100).batch(32).repeat(2)
test_data = test_data.batch(32)



# Now you can train your model
model.fit(train_data, epochs=1, validation_data=test_data, callbacks=[tensorboard_callback])
