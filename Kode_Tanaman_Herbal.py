import os
from dotenv import load_dotenv
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import time
from bert_score import score  # Import BERTScore
import matplotlib.pyplot as plt

# Memuat variabel lingkungan dari file .env
load_dotenv()

API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
instruction = (
    "Gemini, sekarang Anda adalah seorang ahli botani yang memahami tentang tanaman herbal di Papua. "
    "Dalam chat ini, berikan hanya jawaban tentang tanaman herbal yang ada di Papua. "
    "Anda tidak boleh menjelaskan tentang hal yang lain selain tanaman herbal di Papua. "
    "Jika kata yang dimasukkan hanya satu atau dua huruf atau bahkan tidak beraturan, maka Anda harus meminta melengkapi kata."
)
print("Halo! Saya adalah bot yang akan menjelaskan tentang tanaman herbal di Papua")

# Inisialisasi recognizer untuk pengenalan suara
recognizer = sr.Recognizer()

def text_to_speech(text, speed=1.0):
    # Menghilangkan simbol '*'
    text = text.replace('*', '')
    
    # Generate speech using gTTS with speed modification
    tts = gTTS(text=text, lang='id', slow=False)
    tts.save('ngomong.mp3')
    
    # Load generated speech with pydub
    sound = AudioSegment.from_file('ngomong.mp3', format='mp3')
    
    # Play the sound
    play(sound)
    
    # Remove the mp3 file after playing
    os.remove('ngomong.mp3')

def recognize_speech():
    with sr.Microphone() as source:
        print("Virtual Assistant: Mendengarkan...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
        try:
            print("Virtual Assistant: Mengenali...")
            text = recognizer.recognize_google(audio, language='id-ID')
            print(f"Kamu: {text}")
            return text
        except sr.UnknownValueError:
            print("Virtual Assistant: Maaf, saya tidak mengerti.")
        except sr.RequestError as e:
            print(f"Virtual Assistant: Maaf, ada masalah dengan layanan pengenalan suara: {e}")
        return None

# Daftar untuk menyimpan nilai Precision, Recall, dan F1
precision_scores = []
recall_scores = []
f1_scores = []

try:
    while True:
        question = recognize_speech()
        if question is None:
            continue

        try:
            response_text = chat.send_message(instruction + question).text
            response = response_text
            
            # Menghitung BERTScore
            references = ["Informasi tentang tanaman herbal di Papua."]
            candidate = [response_text]
            P, R, F1 = score(candidate, references, lang="id", verbose=True)
            
            precision_score_value = P.item()
            recall_score_value = R.item()
            f1_score_value = F1.item()
            
            precision_scores.append(precision_score_value)
            recall_scores.append(recall_score_value)
            f1_scores.append(f1_score_value)
            
            print(f"BERTScore (Precision): {precision_score_value:.4f}")
            print(f"BERTScore (Recall): {recall_score_value:.4f}")
            print(f"BERTScore (F1): {f1_score_value:.4f}")
        except Exception as e:
            print(f"Terjadi kesalahan saat mengirim pesan: {e}")
            time.sleep(5)  # Tunggu sebelum mencoba lagi
            continue

        print('\n')
        print(f"Bot: {response}")
        print('\n')
        text_to_speech(response, speed=1.5)  # Menyesuaikan kecepatan ke 1.5x dari kecepatan normal
        instruction = ''
except KeyboardInterrupt:
    # Tampilkan grafik setelah loop dihentikan
    plt.figure(figsize=(12, 6))
    
    plt.plot(
    precision_scores, marker='o', label='Precision')
    plt.plot(recall_scores, marker='o', label='Recall')
    plt.plot(f1_scores, marker='o', label='F1')
    
    plt.title('BERTScore (Precision, Recall, F1) dari Respon Bot')
    plt.xlabel('Iterasi')
    plt.ylabel('Skor')
    plt.legend()
    plt.grid(True)
    plt.show()
