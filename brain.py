import ollama
import os
import json
import chromadb
from concurrent.futures import ThreadPoolExecutor
from transcribe import SpeechToText
from therapist_bot import TherapistBot

class TherapistBrain:
    def __init__(self):
        print("--- Initializing Neural Engines & Vector Vault ---")
        self.stt = SpeechToText()
        self.emotion_bot = TherapistBot()
        self.model = "llama3"
        
        # Initialize Vector DB
        os.makedirs("data/vector_vault", exist_ok=True)
        self.db_client = chromadb.PersistentClient(path="data/vector_vault")
        self.collection = self.db_client.get_or_create_collection(name="therapy_sessions")

    def load_history(self):
        # We now use Vector DB instead of JSON, but keeping this for compatibility
        return []

    def check_safety(self, text):
        triggers = ["hurt myself", "suicide", "end my life", "self-harm", "kill myself"]
        return any(trigger in text.lower() for trigger in triggers)

    def get_relevant_context(self, current_input):
        try:
            results = self.collection.query(query_texts=[current_input], n_results=3)
            return "\n".join(results['documents'][0]) if results['documents'] else ""
        except:
            return ""

    def generate_streaming_response(self, audio_path):
        # 1. Parallel STT and Emotion
        with ThreadPoolExecutor() as executor:
            future_text = executor.submit(self.stt.transcribe, audio_path)
            future_emotion = executor.submit(self.emotion_bot.analyze_voice, audio_path)
            transcript = future_text.result()
            emotion = future_emotion.result()

        # 2. Safety Check
        if self.check_safety(transcript):
            yield f"METADATA|{transcript}|{emotion}|"
            yield "I'm concerned about what you're sharing. Please reach out to a professional immediately."
            return

        # 3. Memory Retrieval
        past_memories = self.get_relevant_context(transcript)
        
        # 4. Stream from Ollama
        system_prompt = f"You are SoulSync, a therapist. Current Tone: {emotion}. Relevant History: {past_memories}. Instructions: Be brief (3-4 sentences)."
        
        # Yield metadata first for the UI
        yield f"METADATA|{transcript}|{emotion}|"

        stream = ollama.chat(
            model=self.model,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': transcript}],
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            yield content

        # 5. Save to Vector Vault
        self.collection.add(
            documents=[f"User: {transcript} | AI: {full_response}"],
            ids=[f"id_{os.urandom(4).hex()}"]
        )