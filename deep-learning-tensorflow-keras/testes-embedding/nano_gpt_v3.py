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
import string

import pickle
import pandas as pd

from keras.utils import register_keras_serializable
from keras.layers import Layer, Embedding, Input, Dense, TextVectorization, Flatten, Reshape
from keras.models import Model
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

print(f'Tensorflow Version : {tf.__version__}')
print(f'Keras Version : {keras.__version__}')

# %%
# Parameters
vocab_size = 80000  # Vocabulary size
maxlen = 80  # Maximum length of input sequences
dim_model = 256  # Dimension of the model
num_heads = 8  # Number of attention heads
ff_dim = 512  # Dimension of the feed-forward layer
num_blocks = 4  # Number of transformer blocks
dropout = 0.1  # Dropout rate

batch_size = 128

# clearned_corpus = f'./data/small_20000t_170Ksamples_clearned_corpus.txt.gz'
# file_word_dict = './data/small_20000t_170Ksamples_word_dict.pickle'
# file_count_words = './data/small_20000t_170Ksamples_count_words.parquet'
# model_filename = './models/small_20000t_170Ksamples_nano_gpt_by_marcelo.keras'

clearned_corpus = f'./data/clearned_corpus_00.txt.gz'
file_word_dict = './data/word_dict.pickle'
file_count_words = './data/count_words.parquet'
model_filename = './models/nano_gpt_v3_by_marcelo.keras'

# %%
df_count_words = pd.read_parquet(file_count_words, engine='pyarrow')
df_count_words.info()
vocab = df_count_words.sort_values('count', ascending=False).head(vocab_size - 1)['word'].unique().copy()
del df_count_words
print(vocab.shape)

# %%
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>', filters='\n')
tokenizer.fit_on_texts(vocab)
del vocab

# %%
with gzip.open(clearned_corpus, 'rt') as f:
  lines = f.readlines()

# %%
sequences = tokenizer.texts_to_sequences(lines)
vectorized_sequences = pad_sequences(sequences, padding="pre", maxlen=maxlen + 1, dtype='int64')

print('Length of lines:', len(lines), 'Length of sequences:', len(sequences), 'Length of vectorized sequences:', len(vectorized_sequences))
del sequences

# %%
tokenizer.texts_to_sequences(["na era pré-moderna , foi a forma meditativa trad"])

# %%
print(lines[100], len(lines[100].split()))
print(vectorized_sequences[100], vectorized_sequences[100].shape)

# %%
print(tokenizer.sequences_to_texts([vectorized_sequences[100]]))
print(tokenizer.index_word[13317])

# %%
# random.shuffle(vectorized_sequences)
# train_pairs, test_pairs = train_test_split(text_pairs, test_size=0.10)
num_val_samples = int(0.05 * len(lines))
num_train_samples = len(lines) - 2 * num_val_samples
print(f'Samples:[{len(lines)}] - Train:[{num_train_samples}] - Val:[{num_val_samples}] - Test:[{len(lines) - num_train_samples - num_val_samples}]')
train_pairs = vectorized_sequences[:num_train_samples]
val_pairs = vectorized_sequences[num_train_samples:num_train_samples + num_val_samples]
test_pairs = lines[num_train_samples + num_val_samples:]
del vectorized_sequences
del lines

# %%


def format_dataset(batch_lines):
  # Separando features (X) e labels (y)
  x = batch_lines[:, :-1]  # Todas as colunas, exceto a última
  y = batch_lines[:, 1:]   # A última coluna é o label

  # x.set_shape([None, maxlen])
  # y.set_shape([None, maxlen])

  return x, y


def make_dataset(sequences):
  # dataset = tf.data.Dataset.from_tensor_slices(["brasil o maior país do mundo", "são paulo capital"])
  dataset = tf.data.Dataset.from_tensor_slices(sequences)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
  return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE).cache()


train_ds = make_dataset(train_pairs)

# %%
if True:
  for x, y in train_ds.take(1).cache():
    print('x:', x)
    print('y:', y)
    print('----')

# %%
val_ds = make_dataset(val_pairs)

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
  # reshape = Reshape((-1, 1))(x)  # Output shape: (None, 80 * 40000)
  # flattened = Flatten()(reshape)
  outputs = Dense(vocab_size, activation="softmax")(x)
  return Model(inputs, outputs)

# %%


class TextGeneration(tf.keras.callbacks.Callback):
  def __init__(self, tokenizer, test_pairs, max_length=100, temperature=1.0, model=None):
    super(TextGeneration, self).__init__()
    self.tokenizer = tokenizer
    self.test_pairs = test_pairs
    self.max_length = max_length
    self.temperature = temperature
    self.m = model

  def on_epoch_end(self, epoch, logs=None):
    print(f'\nGenerating text after epoch: {epoch + 1}')
    prompt, label, text_generated = self.auto_generated_text()
    print(f'Label    [{len(label.split()):03}]:[{label}]')
    print(f'Prompt   [{len(prompt.split()):03}]:[{prompt}]')
    print(f'Generated[{len(text_generated.split()):03}]:[{text_generated}]')

  def auto_generated_text(self):
    prompt = random.choice(self.test_pairs)
    split = prompt.split()
    if len(split) > 20:
      split = split[:20]
    prompt = ' '.join(split[:-1])
    label = ' '.join(split[1:])
    text_generated = self.generate_text(prompt)
    return prompt, label, text_generated

  def detokenize(self, tokens):
    result = []
    for token in tokens:
      result.append(self.index_word[token.numpy()])
    return ' '.join(result)

  def generate_text(self, prompt):
    # print(f'Prompt   [{len(prompt.split()):03}]:[{prompt}]')
    input_eval = self.tokenizer.texts_to_sequences([prompt])
    input_eval = pad_sequences(input_eval, maxlen=self.max_length, padding='pre')

    text_generated = prompt.split()
    count_it = 0  # Counter for iterations
    for _ in range(len(prompt.split()) - 1, maxlen):
      count_it += 1
      if self.m is None:
        predictions = self.model(input_eval)
      else:
        predictions = self.m(input_eval)
      predictions = tf.squeeze(predictions, 0)  # Shape: (sequence_length, vocab_size)
      predictions = predictions / self.temperature
      token = tf.argmax(predictions[:, -1], axis=0)
      word_predicted = self.tokenizer.index_word[token.numpy()]

      if word_predicted == '[eos]':
        break

      text_generated.append(word_predicted)
      input_eval = self.tokenizer.texts_to_sequences([' '.join(text_generated)])
      input_eval = pad_sequences(input_eval, maxlen=self.max_length, padding='pre')

    # Join the generated words into a single string
    final_text = ' '.join(text_generated)

    for i in string.punctuation:
      final_text = final_text.replace(' ' + i, i)

    return final_text


# %%
model = build_model(vocab_size, maxlen, dim_model, num_heads, ff_dim, num_blocks, dropout)
loss = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer="adam",
              loss=loss,
              metrics=["accuracy"])

model.summary()

# %%
if os.path.exists(model_filename):
  model = keras.models.load_model(model_filename)
  print(f'Model loaded from {model_filename}')
  print(model)

# %%
checkpoint_callback = ModelCheckpoint(
    filepath=model_filename,
    save_weights_only=False,
    save_freq='epoch')

text_gen_callback = TextGeneration(tokenizer, test_pairs=test_pairs, max_length=maxlen, temperature=1)

model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[checkpoint_callback, text_gen_callback])

# %%
text_gen_callback = TextGeneration(tokenizer, test_pairs=test_pairs, max_length=maxlen, temperature=1, model=model)
for _ in range(5):
  prompt, label, generated = text_gen_callback.auto_generated_text()

  print(f'Prompt[{len(prompt.split())}]:\n{prompt}')
  print(f'Label[{len(label.split())}]:\n{label}')
  print(f'Generated[{len(generated.split())}]:\n{generated}')
  print('----')

# %%
prompt = 'a sequencia de numeros inteiros'  # input('Prompt: ')
text_gen_callback = TextGeneration(tokenizer, test_pairs=test_pairs, max_length=maxlen, temperature=1.0, model=model)
generated = text_gen_callback.generate_text(prompt)
print(f'Prompt[{len(prompt.split())}]:\n{prompt}')
print(f'Generated[{len(generated.split())}]:\n{generated}')

# %%
prompt = 'o espaço expositivo é completamente transparente , fechado com panos de vidro e protegido da incidência solar por painéis de casa barco canoa'
split = prompt.split()

if len(split) > 20:
  split = split[:20]

print(f'Texto: [{' '.join(split)}]')

prompt = ' '.join(split[:-1])
print(f'Prompt: [{prompt}]')

label = ' '.join(split[1:])
print(f'Label: [{label}]')

# %%
text_gen_callback.auto_generated_text()

# %%


def detokenize(tokens):
  result = []
  for token in tokens:
    result.append(index_word[token.numpy()])
  return ' '.join(result)


index_word = dict(zip(range(len(vocab)), vocab))

prompt = 'formação estrelar na grande nuvem de magalhães'  # input('Prompt: ')
len_prompt = len(prompt.split())
print(f'len_prompt: [{len_prompt}] - last_word-1: [{prompt.split()[len_prompt - 1]}] - last_word-2: [{prompt.split()[len_prompt - 2]}]')
# Add [bos] token if not already present
# if '[bos]' not in prompt:
#  prompt = '[bos] ' + prompt

# Tokenize the prompt and add batch dimension
input_eval = vectorization(prompt)
# input_eval = tf.expand_dims(input_eval, 0)  # Shape: (1, sequence_length)

# Initialize list to store generated words
text_generated = prompt.split()
count_it = 0  # Counter for iterations
# Generate text iteratively
for i in range(len_prompt, maxlen):
  count_it += 1
  print(f'Prompt       : [{text_generated}]')
  print(f'Vector Prompt: [{input_eval}]')
  predictions = model(input_eval)  # Use the training model
  # Remove batch dimension and apply temperature scaling
  predictions = tf.squeeze(predictions, 0)  # Shape: (sequence_length, vocab_size)

  predictions = predictions / 1.0
  # print(predictions.shape, predictions)
  tokens = tf.argmax(predictions, axis=1)
  print(f'Predit Tokens: {tokens}')
  print(f'Raw Generated: [{detokenize(tokens)}]')
  token = tokens[i]
  word_predicted = index_word[token.numpy()]
  print(f'Iteration    :[{count_it:03}] - Pos:[{i}] - Token:[{token}] - Word pred:[{word_predicted}]')

  # Stop if the [eos] token is generated
  if word_predicted == '[eos]':
    break

  # Append the predicted word to the generated text
  text_generated.append(word_predicted)

  # Update input for the next iteration
  input_eval = vectorization(' '.join(text_generated))
  # input_eval = tf.expand_dims(input_eval, 0)  # Add batch dimension

# Join the generated words into a single string
final_text = ' '.join(text_generated)
final_text

# %% [markdown]
#
