# %%
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/marcelo/miniconda3/envs/nlp"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# %%
import tensorflow as tf
import keras
import random
import gzip
import numpy as np

import pickle
import pandas as pd

from keras.utils import register_keras_serializable
from keras.layers import Layer, Embedding, Input, Dense, TextVectorization, Flatten, Reshape
from keras.models import Model
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

print(f'Tensorflow Version : {tf.__version__}')
print(f'Keras Version : {keras.__version__}')

# %%
# Parameters
vocab_size = 40000  # Vocabulary size
maxlen = 40  # Maximum length of input sequences
dim_model = 128  # Dimension of the model
num_heads = 2  # Number of attention heads
ff_dim = 512  # Dimension of the feed-forward layer
num_blocks = 2  # Number of transformer blocks
dropout = 0.1  # Dropout rate

batch_size = 256

# clearned_corpus = f'./data/small_20000t_170Ksamples_clearned_corpus.txt.gz'
# file_word_dict = './data/small_20000t_170Ksamples_word_dict.pickle'
# file_count_words = './data/small_20000t_170Ksamples_count_words.parquet'
# model_filename = './models/small_20000t_170Ksamples_nano_gpt_by_marcelo.keras'

clearned_corpus = f'./data/clearned_corpus.txt.gz'
file_word_dict = './data/word_dict.pickle'
file_count_words = './data/count_words.parquet'
model_filename = './models/nano_gpt_by_marcelo.keras'

# %%
df_count_words = pd.read_parquet(file_count_words, engine='pyarrow')
df_count_words.info()
vocab = df_count_words.sort_values('count', ascending=False).head(vocab_size - 2)['word'].unique().copy()
del df_count_words
print(vocab.shape)

# %%
with gzip.open(clearned_corpus, 'rt') as f:
  lines = f.readlines()

text_pairs = []
for line in lines:
  _split = line.split()
  x, y = (' '.join(_split[:-1]), _split[-1])
  text_pairs.append((x, y))

del lines
print('Length of pairs:', len(text_pairs))

# %%
vectorization = TextVectorization(
    vocabulary=vocab,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=maxlen,
    standardize=None,
)
vocab = vectorization.get_vocabulary()
print('Vocab Size:', len(vocab))
vocab = [str(x) for x in vocab]

# %%
vectorization('escultura em barro pintado de um sanfoneiro um dos músicos que integram as bandas de forró caruaru pernambuco o termo forró segundo o filólogo pernambucano evanildo bechara é uma redução de forrobodó que por sua vez')

# Test with: small_20000t_170Ksamples_
# <tf.Tensor: shape=(40,), dtype=int64, numpy=
# array([ 1435,     8,  8691,  7607,     2,    10, 11655,    10,    19,
#       13004,    11,  8568,    23,  1880,     2,  2207,  4661,  1127,
#           4,    91,  2207,    73,     4,  6308,  3455, 19886,     1,
#           9,    12, 15187,     2,  8971,    11,    17,    34,   208,
#           0,     0,     0,     0])>

# Test with: big_40000t_9.8MMsamples_
# <tf.Tensor: shape=(40,), dtype=int64, numpy=
# array([ 3118,     9,  5420,  8115,     2,    11,     1,    11,    20,
#        1641,    16,  9655,    26,   438,     2, 10688,  9693,  1132,
#           6,   208, 10688,    97,     6, 11580,  5089,     1,     1,
#           8,    10,  4351,     2,     1,    16,    17,    30,   309,
#           0,     0,     0,     0])>


# %%
# random.shuffle(text_pairs)
# train_pairs, test_pairs = train_test_split(text_pairs, test_size=0.10)
num_val_samples = int(0.05 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
print(f'Samples:[{len(text_pairs)}] - Train:[{num_train_samples}] - Val:[{num_val_samples}] - Test:[{len(text_pairs) - num_train_samples - num_val_samples}]')
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

# %%
pair = random.choice(text_pairs)
print(f'length: {len(pair[0].split())} - text: [{pair[0]}] - vector:\n{vectorization(pair[0])}')
print(f'length: {len(pair[1].split())} - text: [{pair[1]}] - vector: [{vectorization(pair[1])[0]}]')

# %%
'[end]' in vocab

# %%


def format_dataset(x, y):
  x = vectorization(x)
  y = vectorization(y)[:, 0]
  return x, y


def make_dataset(pairs):
  x, y = zip(*pairs)
  x = list(x)
  y = list(y)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
  # return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE).cache()
  return dataset.prefetch(tf.data.AUTOTUNE).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

# %%
if False:
  for final_text in train_ds.take(1).cache():
    print(final_text)

# %%


@register_keras_serializable()
class PositionalEmbedding(Layer):
  def __init__(self, vocab_size, dim_model, max_len, **kwargs):
    super(PositionalEmbedding, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.dim_model = dim_model
    self.max_len = max_len
    # self.token_emb = Embedding(input_dim=vocab_size, output_dim=d_model)
    # self.pos_emb = Embedding(input_dim=max_len, output_dim=d_model)

  def build(self, input_shape):
    # Initialize the token embedding layer
    self.token_emb = Embedding(input_dim=self.vocab_size, output_dim=self.dim_model)
    # Initialize the positional embedding layer
    self.pos_emb = Embedding(input_dim=self.max_len, output_dim=self.dim_model)
    # Mark the layer as built
    super(PositionalEmbedding, self).build(input_shape)

  def call(self, x):
    max_len = tf.shape(x)[-1]
    positions = tf.range(start=0, limit=max_len, delta=1)
    positions = self.pos_emb(positions)
    x = self.token_emb(x)
    return x + positions

  def get_config(self):
    config = super(PositionalEmbedding, self).get_config()
    config.update({
        'vocab_size': self.vocab_size,
        'dim_model': self.dim_model,
        'max_len': self.max_len
    })
    return config

# %%


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
  x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
  x = Dropout(dropout)(x)
  x = LayerNormalization(epsilon=1e-6)(x)
  res = x + inputs

  x = Dense(ff_dim, activation="relu")(res)
  x = Dropout(dropout)(x)
  x = Dense(inputs.shape[-1])(x)
  x = LayerNormalization(epsilon=1e-6)(x)
  return x + res


def build_model(vocab_size, max_len, dim_model, num_heads, ff_dim, num_blocks, dropout=0):
  inputs = Input(shape=(max_len,))
  x = PositionalEmbedding(vocab_size, dim_model, max_len)(inputs)
  for _ in range(num_blocks):
    x = transformer_block(x, dim_model, num_heads, ff_dim, dropout)
  # Flatten the input
  reshape = Reshape((-1, 1))(x)  # Output shape: (None, 80 * 40000)
  flattened = Flatten()(reshape)
  outputs = Dense(vocab_size, activation="softmax")(flattened)
  # outputs = Reshape((1, vocab_size))(dense)
  return Model(inputs, outputs)


# %%
# Build and compile the model
model = build_model(vocab_size, maxlen, dim_model, num_heads, ff_dim, num_blocks, dropout)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Summary of the model
model.summary()

# %%


class TextGeneration(tf.keras.callbacks.Callback):
  def __init__(self, tokenizer, test_pairs, max_length=100, temperature=1.0, model=None):
    super(TextGeneration, self).__init__()
    self.tokenizer = tokenizer
    self.test_pairs = test_pairs
    self.max_length = max_length
    self.temperature = temperature
    self.vocab = self.tokenizer.get_vocabulary()
    self.index_word = dict(zip(range(len(self.vocab)), self.vocab))
    self.m = model

  def on_epoch_end(self, epoch, logs=None):
    print(f'\nGenerating text after epoch: {epoch + 1}')
    prompt, label, text_generated = self.auto_generated_text()
    print(f'Label    [{len(label.split()):03}]:[{label}]')
    print(f'Prompt   [{len(prompt.split()):03}]:[{prompt}]')
    print(f'Generated[{len(text_generated.split()):03}]:[{text_generated}]')

  def auto_generated_text(self):
    prompt, label = random.choice(self.test_pairs)
    text_generated = self.generate_text(prompt)
    return prompt, label, text_generated

  def generate_text(self, prompt):
    input_eval = self.tokenizer(prompt)
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = prompt.split()
    for i in range(self.max_length):
      if self.m is None:
        predictions = self.model(input_eval)
      else:
        predictions = self.m(input_eval)
      predictions = predictions / self.temperature
      predicted_id = np.argmax(predictions[0])
      word_predicted = self.index_word[predicted_id]
      if word_predicted == '[end]':
        break
      text_generated.append(word_predicted)
      input_eval = vectorization(' '.join(text_generated))
      input_eval = tf.expand_dims(input_eval, 0)

    final_text = ' '.join(text_generated)

    return final_text


# %%
if os.path.exists(model_filename):
  model = keras.models.load_model(model_filename)
  print(f'Model loaded from {model_filename}')
  print(model)

# %%
checkpoint_callback = ModelCheckpoint(
    filepath='./nano_gpt_by_marcelo.keras',
    save_weights_only=False,
    save_freq='epoch')

# Create the TextGeneration callback
text_gen_callback = TextGeneration(vectorization, test_pairs=test_pairs, max_length=maxlen, temperature=1)

# Train the model with the callback
model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[checkpoint_callback, text_gen_callback])

# %%
text_gen_callback = TextGeneration(vectorization, test_pairs=test_pairs, max_length=maxlen, temperature=1, model=model)
for _ in range(5):
  prompt, label, generated = text_gen_callback.auto_generated_text()

  print(f'Prompt[{len(prompt.split())}]:\n{prompt}')
  print(f'Label[{len(label.split())}]:\n{label}')
  print(f'Generated[{len(generated.split())}]:\n{generated}')
  print('----')

# %%
prompt = 'a capital de portugal é'  # input('Prompt: ')
text_gen_callback = TextGeneration(vectorization, test_pairs=test_pairs, max_length=maxlen, temperature=1.0, model=model)
generated = text_gen_callback.generate_text(prompt)
print(f'Prompt[{len(prompt.split())}]:\n{prompt}')
print(f'Generated[{len(generated.split())}]:\n{generated}')

"""
Prompt[5]:
a capital de portugal é
Generated[39]:
a capital de portugal é sede do país que tem chamada de sua pico com a área de frequência de pressão o pico ao do pico ao de direito [UNK] o termo oficial de teologia da o direito de
"""
