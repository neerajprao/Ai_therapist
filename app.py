from flask import Flask, render_template, request, jsonify
from brain import TherapistBrain
import os

app = Flask(__name__)

# Initialize the brain once so models stay in RAM
print("Initializing AI Therapist Brain...")
brain = TherapistBrain()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    
    # Save the incoming audio from the browser
    temp_path = "data/raw_audio/web_capture.wav"
    audio_file.save(temp_path)
    
    # Run the full pipeline (Transcription -> Emotion -> LLM)
    text, emotion, response = brain.generate_response(temp_path)
    
    return jsonify({
        "user_said": text,
        "detected_emotion": emotion,
        "ai_response": response
    })

if __name__ == '__main__':
    # Ensure the audio directory exists
    os.makedirs("data/raw_audio", exist_ok=True)
    
    # Debug=False is important on Macs so it doesn't load the model twice
    app.run(port=5000, debug=False)