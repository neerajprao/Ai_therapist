import os
import uuid
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
from dotenv import load_dotenv

# 1. Load secrets from .env file
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 2. Initialize Neural Brain (Whisper + Llama-3 + Emotion2Vec)
from brain import TherapistBrain
brain = TherapistBrain()

print("--- SoulSync Secure Mode: Deepgram Cloud Voice Active ---")

@app.route('/')
def index():
    session['id'] = str(uuid.uuid4())
    os.makedirs("static/audio", exist_ok=True)
    os.makedirs("data/raw_audio", exist_ok=True)
    return render_template('index.html')

@app.route('/process_audio_stream', methods=['POST'])
def process_audio_stream():
    if 'audio' not in request.files: return "No audio", 400
    audio_file = request.files['audio']
    temp_path = f"data/raw_audio/{session['id']}_input.wav"
    audio_file.save(temp_path)
    
    def generate():
        # Streams text from Llama-3 locally via brain.py
        for chunk in brain.generate_streaming_response(temp_path):
            yield chunk
            
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/get_audio', methods=['POST'])
def get_audio():
    data = request.json
    text = data.get('text', '')
    emotion = data.get('emotion', 'Neutral')
    
    # Using 'aura-luna-en' for a warm, human therapist voice
    voice_model = "aura-luna-en" 
    
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("static/audio", filename)

    try:
        url = f"https://api.deepgram.com/v1/speak?model={voice_model}"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"text": text}

        # Request audio from Deepgram Aura-2
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            return jsonify({"audio_url": f"/static/audio/{filename}"})
        elif response.status_code == 401:
            return jsonify({"error": "Unauthorized: Check your Deepgram API Key in .env"}), 401
        else:
            return jsonify({"error": f"Deepgram Error: {response.text}"}), 500

    except Exception as e:
        print(f"System Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # threaded=True lets the UI remain snappy while the voice is being fetched
    app.run(port=5000, debug=False, threaded=True)