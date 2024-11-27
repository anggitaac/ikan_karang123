from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os

app = Flask(__name__)

# Load model dengan format .keras
model = tf.keras.models.load_model('1.keras')

# Fungsi untuk memproses gambar
def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).resize((150, 150))  # Resize gambar ke ukuran yang sesuai dengan model
    image = np.array(image) / 255.0  # Normalisasi gambar ke rentang [0, 1]
    return np.expand_dims(image, axis=0)  # Menambah dimensi untuk batch size

# Fungsi untuk mendecode hasil prediksi
def decode_prediction(prediction, threshold=0.7):
    labels = [ 
        "Clownfish", "Yellow Tang", "Palette Surgeonfish", 
        "Three-stripe Damselfish", "Moorish Idol", 
        "Acanthurus Bahianus", "Pomacanthus Annularis", "Raccoon Butterflyfish"
    ]  # Label yang sesuai dengan kelas model Anda
    
    predicted_index = np.argmax(prediction)  # Dapatkan indeks dengan probabilitas tertinggi
    confidence = np.max(prediction)  # Ambil confidence tertinggi
    
    # Jika confidence di bawah threshold, kembalikan "Tidak Diketahui"
    if confidence < threshold:
        return "Tidak Diketahui", confidence
    else:
        return labels[predicted_index], confidence

@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # Mengambil file gambar dari request
        file = request.files['image']
        
        # Memproses gambar dan prediksi
        image_data = file.read()  # Membaca data gambar
        image = preprocess_image(image_data)  # Memproses gambar
        prediction = model.predict(image)  # Membuat prediksi dengan model
        
        # Decode prediksi menjadi label yang lebih mudah dipahami
        label, confidence = decode_prediction(prediction)
        
        return jsonify({
            'prediction': label, 
            'confidence': float(confidence),  # Convert numpy float ke tipe data Python
            'message': "Jenis ikan berhasil dikenali." if label != "Tidak Diketahui" else "Jenis ikan tidak ditemukan."
        })  # Mengembalikan hasil prediksi
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Menjalankan aplikasi pada port yang ditentukan di Railway atau pada 5000 sebagai default
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
