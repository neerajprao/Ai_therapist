import os
import sys
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
import numpy
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session
import uuid
import scipy.io.wavfile as wavfile

# 1. Monkeypatch and Device Setup
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from bark import SAMPLE_RATE, generate_audio, preload_models
device = "mps" if torch.backends.mps.is_available() else "cpu"

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 2. Brain Initialization
from brain import TherapistBrain
brain = TherapistBrain()

print(f"--- POWER MODE: {device.upper()} ---")
# Using smaller models for speed
preload_models(
    text_use_gpu=(device == "mps"),
    text_use_small=True,
    coarse_use_small=True,
    fine_use_small=True,
)

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
    
    # CRITICAL: Limit text length for Bark to keep it fast
    # If the AI rambles, Bark takes exponentially longer.
    short_text = text.split('.') [0] + "." # Just process the first sentence for instant feedback

    voice_preset = "v2/en_speaker_9"
    processed_text = short_text
    if "Sad" in emotion: processed_text = f"[sighs] {short_text}"
    elif "Happy" in emotion: processed_text = f"♪ {short_text} ♪"

    filename = f"bark_{uuid.uuid4().hex}.wav"
    filepath = os.path.join("static/audio", filename)

    try:
        # Generate with lower precision/smaller steps if possible
        audio_array = generate_audio(processed_text, history_prompt=voice_preset, silent=True)
        wavfile.write(filepath, SAMPLE_RATE, audio_array)
        return jsonify({"audio_url": f"/static/audio/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False, threaded=True)