import ollama
import os
from transcribe import SpeechToText
from therapist_bot import TherapistBot

class TherapistBrain:
    def __init__(self):
        # Initialize our local "Ears" and "Translator"
        self.stt = SpeechToText()
        self.emotion_bot = TherapistBot()
        self.model = "llama3"
        
        # This list will store the conversation history
        # Format: [{'role': 'user/assistant', 'content': '...'}]
        self.history = [] 

    def generate_response(self, audio_path):
        # 1. Get the "What" (Transcription)
        transcript = self.stt.transcribe(audio_path)
        
        # 2. Get the "How" (Emotion Detection)
        emotion = self.emotion_bot.analyze_voice(audio_path)
        
        # 3. Define the Emotional System Prompt
        # This is injected every time to keep the AI in "Therapist Mode"
        system_prompt = f"""
        You are a professional, empathetic therapist. 
        The user is currently speaking with a {emotion} tone.
        
        INSTRUCTIONS:
        - Acknowledge their emotional state ({emotion}) naturally.
        - If they sound sad or anxious, be validating and gentle.
        - Reference previous things they've said in this session to show you are listening.
        - Keep your response under 4-5 sentences to maintain a conversational flow.
        """
        
        # 4. Update History with the new User Message
        # We tag the user message with their tone so the LLM "sees" it
        self.history.append({
            'role': 'user', 
            'content': f"[Tone: {emotion}] {transcript}"
        })
        
        # 5. Build the full message list (System + History)
        messages = [{'role': 'system', 'content': system_prompt}] + self.history
        
        print(f"\n[Brain] Analyzing {emotion} audio...")
        
        # 6. Generate response via Ollama
        try:
            response = ollama.chat(model=self.model, messages=messages)
            ai_reply = response['message']['content']
            
            # Save the AI's reply to history for the next turn
            self.history.append({'role': 'assistant', 'content': ai_reply})
            
            return transcript, emotion, ai_reply
            
        except Exception as e:
            return transcript, emotion, f"Error calling Ollama: {str(e)}"

if __name__ == "__main__":
    brain = TherapistBrain()
    
    # --- MULTI-TURN TEST ---
    # Call 1: The Initial Audio
    audio_1 = "data/raw_audio/sad.wav"
    if os.path.exists(audio_1):
        t1, e1, r1 = brain.generate_response(audio_1)
        print(f"\n--- TURN 1 ---")
        print(f"User: {t1} (Tone: {e1})")
        print(f"AI: {r1}")

        # Call 2: The Follow-up (Record a 3-second audio saying "I don't know what to do.")
        audio_2 = "data/raw_audio/followup.wav"
        if os.path.exists(audio_2):
            t2, e2, r2 = brain.generate_response(audio_2)
            print(f"\n--- TURN 2 ---")
            print(f"User: {t2} (Tone: {e2})")
            print(f"AI: {r2}")
        else:
            print("\n[!] Please record 'data/raw_audio/followup.wav' to test memory.")
    else:
        print(f"[!] {audio_1} not found.")