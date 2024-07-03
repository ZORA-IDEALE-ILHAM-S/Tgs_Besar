import os
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play

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

def text_to_speech(text):
    # Menghilangkan simbol '*'
    text = text.replace('*', '')
    
    # Generate speech
    tts = gTTS(text=text, lang='id')
    tts.save('ngomong.mp3')
    
    # Load generated speech with pydub
    sound = AudioSegment.from_file('ngomong.mp3', format='mp3')
    
    # Play the sound
    play(sound)

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
        except sr.RequestError:
            print("Virtual Assistant: Maaf, ada masalah dengan layanan pengenalan suara.")
        return None

while True:
    question = recognize_speech()
    if question is None:
        continue

    else:
        response_text = chat.send_message(instruction + question).text
        response = response_text

    print('\n')
    print(f"Bot: {response}")
    print('\n')
    text_to_speech(response)
    instruction = ''
