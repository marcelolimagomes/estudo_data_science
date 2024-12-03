import numpy as np
import tensorflow as tf
from scipy.signal import resample
import streamlit as st
import tensorflow_hub as hub
import librosa

@st.cache_resource 
def load_model():
    model = tf.keras.models.load_model('meu_modelo.keras')
    return model

@st.cache_resource
def load_yamnet_model():
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yamnet_model


def load_audio(file_path):
    """
    Carrega um arquivo de áudio MP3 e retorna o waveform e a taxa de amostragem.

    Args:
    file_path (str): Caminho para o arquivo de áudio MP3.

    Returns:
    tuple: waveform (np.ndarray), sample_rate (int)
    """
    waveform, sample_rate = librosa.load(file_path, sr=None)
    return waveform, sample_rate

def process_and_extract_embeddings(waveform, sample_rate, max_length=16000):
    # Função de resampling usando SciPy
    def scipy_resample(wav, sample_rate):
        if sample_rate != 16000:
            wav = resample(wav, int(16000 / sample_rate * len(wav)))
        return wav

    # Usar tf.py_function para envolver a operação de resampling
    wav = tf.py_function(scipy_resample, [waveform, sample_rate], tf.float32)
    
    # Adicionar padding ou cortar os sinais de áudio
    audio_length = tf.shape(wav)[0]
    if audio_length > max_length:
        wav = wav[:max_length]
    else:
        pad_length = max_length - audio_length
        paddings = [[0, pad_length]]
        wav = tf.pad(wav, paddings, "CONSTANT")
    
    wav = tf.reshape(wav, [max_length])
    yamnet_model = load_yamnet_model()
    # Extrair embeddings usando YAMNet
    scores, embeddings, spectrogram = yamnet_model(wav)
    return embeddings


def predict_audio_class(file_path, model, class_map):
    # Carregar e processar o arquivo de áudio MP3
    waveform, sample_rate = load_audio(file_path)
    
    # Processar e extrair embeddings
    embeddings = process_and_extract_embeddings(waveform, sample_rate)
    
    # Fazer a inferência com o modelo treinado
    predictions = model.predict(embeddings)
    
    # Agregar as previsões (por exemplo, usando a média das previsões)
    final_prediction = np.mean(predictions, axis=0)
    predicted_class_index = np.argmax(final_prediction)
    
    # Mapear o índice previsto para o nome da classe
    predicted_class_name = class_map[predicted_class_index]
    
    return predicted_class_name

def main():
    st.title("Classificador de Áudio")
    st.write("Carregue um arquivo de áudio para classificar")

    uploaded_file = st.file_uploader("Carregar arquivo de áudio", type=["mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")

        mapeamento = {'dog': 0, 'door_wood_creaks': 1, 'glass_breaking': 2}
        mapeamento_inverso = {v: k for k, v in mapeamento.items()}
        model = load_model()
        # Classificar o arquivo de áudio
        predicted_class = predict_audio_class(uploaded_file, model, mapeamento_inverso)
        st.write(f"Classe prevista: {predicted_class}")

if __name__ == "__main__":
    main()