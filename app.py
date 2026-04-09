import os
import uuid
import requests
import base64
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
from dotenv import load_dotenv

load_dotenv()
INWORLD_KEY = os.getenv("INWORLD_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

from brain import TherapistBrain
brain = TherapistBrain()

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
    # Clean text: remove asterisks or brackets if Llama-3 slips up
    text = data.get('text', '').replace('*', '').replace('[', '').replace(']', '')
    emotion = data.get('emotion', 'Neutral')
    
    # --- LUNA EMPATHY ENGINE CONFIG ---
    speed = 1.0
    temperature = 1.1 # Default 'human' variation
    
    if any(e in emotion for e in ["Sad", "Depressed", "Grief"]):
        # [sad] lowers pitch, [whispering] adds breathiness, [sigh] is a non-verbal exhale
        prompt = f"[sad] [whispering] [sigh] {text}"
        speed = 0.82        # Slower for gravity
        temperature = 1.4   # More emotional variation
    elif "Anxious" in emotion:
        prompt = f"[soothing] [breathe] {text}"
        speed = 0.88
        temperature = 0.7   # Stable and calming
    else:
        prompt = f"[calm] {text}"
        speed = 0.95

    filename = f"luna_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("static/audio", filename)

    try:
        # 2026 Inworld TTS-1.5 Max Endpoint
        url = "https://api.inworld.ai/tts/v1/voice"
        headers = {
            "Authorization": f"Basic {INWORLD_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": prompt,
            "voiceId": "Luna",
            "modelId": "inworld-tts-1.5-max",
            "speed": speed,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            res_data = response.json()
            # Decode the base64 audio back into a playable file
            audio_bytes = base64.b64decode(res_data['audioContent'])
            with open(filepath, "wb") as f:
                f.write(audio_bytes)
            return jsonify({"audio_url": f"/static/audio/{filename}"})
        else:
            return jsonify({"error": f"Inworld Error: {response.text}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False, threaded=True)