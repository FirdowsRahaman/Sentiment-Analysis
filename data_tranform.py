import pandas as pd
import tensorflow as tf

from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_string_ops


USE_COLUMNS = ['Text', 'Score']
CLASSES = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
VOCAB_SIZE = 2**13
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIMS = 64  
NUM_CATEGORY = 5
OUTPUT_DIR = '/tmp'
VOCAB_FILE_PATH = os.path.join(OUTPUT_DIR, 'vocab.txt')
DEFAULT_STRIP_REGEX = r'[’!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'


def load_data(dataframe):
  dataframe = pd.read_csv(data_path, usecols=USE_COLUMNS, dtype=str)
  dataframe = dataframe.dropna()
  return (list(dataframe['Text']), np.array(dataframe['Score'].map(CLASSES)))


def _preprocess(self, inputs):
    if ragged_tensor.is_ragged(inputs):
      lowercase_inputs = ragged_functional_ops.map_flat_values(
          gen_string_ops.string_lower, inputs)
      lowercase_inputs = array_ops.identity(lowercase_inputs)
    else:
      lowercase_inputs = gen_string_ops.string_lower(inputs)
    inputs = string_ops.regex_replace(lowercase_inputs, DEFAULT_STRIP_REGEX,
                                        " ")
    tokens = ragged_string_ops.string_split_v2(inputs)
    return tokens


def text_transform(text):
  token = _preprocess(text)
  words = token.to_tensor(shape=(None, MAX_SEQUENCE_LENGTH))
  init = tf.lookup.TextFileInitializer(filename=VOCAB_FILE_PATH, 
                              key_dtype=tf.string, 
                              key_index=0,#tf.lookup.TextFileIndex.WHOLE_LINE,
                              value_dtype=tf.int64, 
                              value_index=1,#tf.lookup.TextFileIndex.LINE_NUMBER,
                              vocab_size=None,
                              delimiter=',')
  
  table = tf.lookup.StaticVocabularyTable(initializer=init, num_oov_buckets=1)             
  word2numbers = table.lookup(words)
  return (word2numbers)