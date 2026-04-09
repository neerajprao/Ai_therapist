import os
import uuid
import requests
import json
import base64
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
INWORLD_KEY = os.getenv("INWORLD_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 2. Initialize Neural Brain
from brain import TherapistBrain
brain = TherapistBrain()

print("--- SoulSync System: Luna Empathy Engine (2026) ---")

@app.route('/')
def index():
    session['id'] = str(uuid.uuid4())
    os.makedirs("static/audio", exist_ok=True)
    return render_template('index.html')

@app.route('/process_audio_stream', methods=['POST'])
def process_audio_stream():
    if 'audio' not in request.files: return "No audio", 400
    audio_file = request.files['audio']
    temp_path = f"data/raw_audio/{session['id']}_input.wav"
    audio_file.save(temp_path)
    
    def generate():
        for chunk in brain.generate_streaming_response(temp_path):
            yield chunk
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/get_audio', methods=['POST'])
def get_audio():
    data = request.json
    text = data.get('text', '')
    emotion = data.get('emotion', 'Neutral')
    
    # --- DYNAMIC EMPATHY SETTINGS ---
    # Default settings (Neutral/Calm)
    speed = 0.95
    temperature = 0.5
    tags = "[calm]"

    # Adjusting based on user's emotional state
    if "Sad" in emotion or "Depressed" in emotion:
        speed = 0.82        # Slower for empathy
        temperature = 0.9   # Higher variation for "cracked" or soft voice
        tags = "[sad] [whispering] [sigh]"
    elif "Anxious" in emotion:
        speed = 0.90        # Slightly slow but steady
        temperature = 0.4   # Very stable/calming
        tags = "[soothing] [breathe]"
    elif "Angry" in emotion:
        speed = 1.05        # Slightly faster but firm
        temperature = 0.2   # Cold/Steady
        tags = "[stern]"

    full_prompt = f"{tags} {text}"
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("static/audio", filename)

    try:
        # Inworld 2026 TTS REST API
        url = "https://api.inworld.ai/tts/v1/voice"
        headers = {
            "Authorization": f"Basic {INWORLD_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": full_prompt,
            "voiceId": "Luna", # Use the 'Meditation' preset
            "modelId": "inworld-tts-1.5-max",
            "speed": speed,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            # Inworld API returns JSON with base64 audioContent
            res_data = response.json()
            audio_data = base64.b64decode(res_data['audioContent'])
            
            with open(filepath, "wb") as f:
                f.write(audio_data)
            return jsonify({"audio_url": f"/static/audio/{filename}"})
        else:
            print(f"Inworld Error: {response.text}")
            return jsonify({"error": "Inworld API failed"}), 500

    except Exception as e:
        print(f"System Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False, threaded=True)