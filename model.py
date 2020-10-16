import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


VOCAB_SIZE = 2**13
EMBEDDING_DIMS = 64
NUM_CATEGORY = 5


def build_model():
  text_input = keras.Input(shape=(None,), name='text', dtype='int64')
  x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIMS, input_length=MAX_SEQUENCE_LENGTH)(text_input)
  x = layers.Conv1D(64, 5, activation='relu')(x)
  x = layers.Bidirectional(layers.LSTM(128))(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  out = layers.Dense(NUM_CATEGORY, activation='sigmoid')(x)
  model = keras.Model(inputs=text_input, outputs=out)
  
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model