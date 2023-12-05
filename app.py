import os
from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import demoji
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
model.load_weights("model/model_weights.h5")  # Update with the correct path

# Fungsi untuk melakukan tahap preprocessing
def preprocess_text(text):
    # Inisialisasi spaCy
    nlp = spacy.load("en_core_web_sm")

    # Pembersihan teks: Hapus karakter khusus dan tanda baca
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    # Pengonversi ke huruf kecil
    text = text.lower()

    # Pemisahan kata menggunakan spaCy
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Lematisasi
    lemmatized_tokens = [token.lemma_ for token in doc]

    # Menghapus emoji
    text = demoji.replace(text, "")

    # Gabungkan kembali token menjadi teks
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# Fungsi untuk membuat Diagram Pie dan menyimpannya di folder static
def generate_pie_chart_and_save(labels, sizes):
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Simpan gambar Pie Chart di folder static
    plt.savefig("static/piechart.png")

# Rute untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', results=None)

# Rute untuk halaman hasil analisis
@app.route('/results', methods=['POST'])
def results():
    if 'file' in request.files:
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        if file:
            # Read CSV file
            data = pd.read_csv(file)

            # Ensure the CSV file has the "review" column
            if 'review' not in data.columns:
                return render_template('index.html', error='CSV file must have a "review" column.')

            texts = data["review"].tolist()
    else:
        user_input = request.form['user_input']
        texts = [user_input]

    # Preprocessing
    user_inputs = [preprocess_text(input_text) for input_text in texts]

    # Tokenisasi dan encoding
    encoded_dict = tokenizer(
        user_inputs,
        add_special_tokens=True,
        max_length=64,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="tf",
    )

    # Prediksi sentimen
    predictions = model.predict([encoded_dict["input_ids"], encoded_dict["attention_mask"]])[0]

    # Make sure the prediction has the expected shape
    if predictions.shape[0] == 1:
        predictions = predictions[0]

    tf_batch = tokenizer(
        user_inputs,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    labels = tf.argmax(tf_predictions, axis=1).numpy()

    # Output klasifikasi
    results = []
    positive_text = ""
    negative_text = ""

    for i, label in enumerate(labels):
        confidence_pos = predictions[i, 1]
        confidence_neg = predictions[i, 0]
        label_text = "Positif" if label == 1 else "Negatif"
        results.append((user_inputs[i], label_text, confidence_pos, confidence_neg))

        # Tambahkan teks ke variabel positif atau negatif sesuai label
        if label == 1:
            positive_text += user_inputs[i] + " "
        else:
            negative_text += user_inputs[i] + " "

    # Membuat Word Cloud terpisah untuk kata-kata positif dan kata-kata negatif yang benar-benar unik
    unique_positive_words = set(positive_text.split())
    unique_negative_words = set(negative_text.split())

    # Filter kata-kata yang hanya muncul dalam ulasan positif
    unique_positive_words_only = unique_positive_words - unique_negative_words

    # Filter kata-kata yang hanya muncul dalam ulasan negatif
    unique_negative_words_only = unique_negative_words - unique_positive_words

    # Hapus gambar Word Cloud sebelumnya (jika ada)
    try:
        os.remove("static/wordcloud_positive.png")
        os.remove("static/wordcloud_negative.png")
    except FileNotFoundError:
        pass

    # Buat Word Cloud untuk kata-kata positif yang benar-benar unik
    wordcloud_positive = WordCloud(width=400, height=400, background_color="white").generate(
        " ".join(unique_positive_words_only))
    wordcloud_positive.to_file("static/wordcloud_positive.png")

    # Buat Word Cloud untuk kata-kata negatif yang benar-benar unik
    wordcloud_negative = WordCloud(width=400, height=400, background_color="white").generate(
        " ".join(unique_negative_words_only))
    wordcloud_negative.to_file("static/wordcloud_negative.png")

    # Membuat Diagram Pie dari hasil klasifikasi
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    pie_labels = ["Positif", "Negatif"]
    pie_sizes = [label_counts[np.where(unique_labels == 1)][0], label_counts[np.where(unique_labels == 0)][0]]
    generate_pie_chart_and_save(pie_labels, pie_sizes)

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
