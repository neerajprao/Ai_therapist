import os
import uuid
import json
import subprocess
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Directory Setup
os.makedirs("static/audio", exist_ok=True)
os.makedirs("data/raw_audio", exist_ok=True)
os.makedirs("data/history", exist_ok=True)

from brain import TherapistBrain
brain = TherapistBrain()

@app.route('/')
def index():
    session['id'] = str(uuid.uuid4())
    history_file = f"data/history/{session['id']}.json"
    initial_data = {
        "session_id": session['id'],
        "history": [{"role": "system", "content": brain.system_prompt}]
    }
    with open(history_file, 'w') as f:
        json.dump(initial_data, f, indent=4)
    return render_template('index.html')

@app.route('/process_audio_stream', methods=['POST'])
def process_audio_stream():
    if 'audio' not in request.files: return "No audio", 400
    audio_file = request.files['audio']
    session_id = session.get('id')
    temp_path = f"data/raw_audio/{session_id}_input.wav"
    audio_file.save(temp_path)
    
    user_text = brain.transcribe_audio(temp_path)
    detected_emotion = brain.detect_emotion_hybrid(user_text, temp_path)
    
    def generate():
        yield f"METADATA|{user_text}|{detected_emotion}|"
        # CHANGED: Pass the detected_emotion to the LLM generator
        for chunk in brain.generate_streaming_response(user_text, session_id, detected_emotion):
            yield chunk
            
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/get_audio', methods=['POST'])
def get_audio():
    data = request.json
    text = data.get('text', '').replace('*', '').replace('[', '').replace(']', '').replace('"', "'")
    
    filename = f"samantha_{uuid.uuid4().hex}.m4a"
    filepath = os.path.join("static/audio", filename)

    try:
        # Changed to Samantha for consistency with macOS voice
        print(f"Synthesizing locally with macOS 'say': '{text[:30]}...'")
        
        subprocess.run([
            'say', 
            '-v', 'Samantha', 
            '-r', '150', 
            '-o', filepath, 
            text
        ], check=True)

        if os.path.exists(filepath):
            return jsonify({"audio_url": f"/static/audio/{filename}"})
        else:
            return jsonify({"error": "TTS failed"}), 500

    except Exception as e:
        print(f"!!! TTS CRASH: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False, threaded=True)