import gradio as gr
import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
model = tf.keras.models.load_model("sentiment_analysis.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen=30
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    prediction = model.predict(padded)[0][0]
    # sentiment = "Positive 😊" if prediction > 1.5 elif "Negative 😠"
    if prediction > 0 and prediction < 0.7:
        sentiment = "Negative 😠"
    elif prediction >= 0.7 and prediction < 1.5:
        sentiment = "Neutral"
    else:
        sentiment = "Positive 😊"
    return f"{sentiment} ({prediction:.2f})"

iface = gr.Interface(fn=predict_sentiment,
                     inputs=gr.Textbox(lines=2, placeholder="Enter a sentence..."),
                     outputs="text",
                     title="LSTM Sentiment Analysis",
                     description="Enter a sentence to analyze sentiment using an LSTM model.")

if __name__ == "__main__":
    iface.launch()