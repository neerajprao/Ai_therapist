# Changelog

## v2.0.0 (unreleased)

Measured results for this version are in [RESULTS.md](RESULTS.md). The issues it addresses are listed in [AUDIT.md](AUDIT.md).

### Fixed

- **Emotion2Vec contributed nothing.** Its bilingual labels (`生气/angry`, `恐惧/fearful`, ...) never matched the lookup, so every score was dropped and the "60/40 ensemble" was Wav2Vec2 alone. Labels are now mapped (`fearful` to `anxious`), an unknown label raises an error, and a regression test covers it.
- **The app could not start.** The LLM was hardcoded to `llama3`. It is now set by `LLM_MODEL` and defaults to `qwen2.5:7b-instruct`.
- **Speaker leakage in evaluation.** The old split was by file over the same 24 actors. The new split shares no speaker between train, validation and test.
- **Wrong labels in the dataset.** SAVEE clips were filed under the wrong class folders; labels now come from filenames.
- **Shared history when the cookie was missing.** Requests without a valid signed user cookie are rejected.
- **Silent fallback to Neutral.** A missing emotion model now shows as "Unavailable" in the UI and in `/health`; errors are logged.
- **Reply text appeared only after audio started.** It now renders as it streams.
- **Fragile stream parsing.** The header can arrive split across chunks, `|` is stripped from transcripts, and multi-byte characters decode correctly.
- **Stage directions leaking through.** `*sighs*` split across streamed chunks is now removed.
- **Train/serve mismatch.** Training, evaluation and inference share one preprocessing function.

### Added

- FastAPI backend with Pydantic models, `/health`, and the same `METADATA|transcript|emotion|` streaming contract.
- Long-term memory on ChromaDB: per-user filtering, current-session exclusion, a distance threshold, and a "Forget me" button backed by `DELETE /memory`.
- Persistent user identity: a signed, HTTP-only, one-year cookie. Each page load still starts a new session.
- Crisis-language safety layer that bypasses the LLM and returns helpline information, plus a fixed doctor-referral line for medication and diagnosis questions.
- Coping-skills knowledge base of ten reviewed general-wellbeing notes (on by default; `KNOWLEDGE_ENABLED=false` turns it off).
- Per-user rate limit on recordings.
- Automated reply-quality evaluation (LLM-as-judge plus keyword rule checks).
- Disclaimer in the UI and a per-mode statement of what data is stored and what leaves the device.
- Local mode (Ollama, local Whisper) and cloud mode (Groq LLM and speech-to-text, ONNX INT8 emotion model, no PyTorch).
- ONNX export with dynamic INT8 quantisation.
- Reproducible training pipeline: inventory, speaker-independent split, cached waveforms, resumable training with early stopping.
- Evaluation suite that writes JSON to `eval/results/` and generates `RESULTS.md`.
- pytest suite using fakes (no model downloads) and a GitHub Actions workflow.
- Environment-driven configuration with `.env.example`, logging, pinned requirements.

### Changed

- Speech output uses the browser's Web Speech API by default. macOS `say` remains as `TTS_BACKEND=say`.
- Chat history sent to the LLM is limited to the last `HISTORY_MAX_TURNS` exchanges.
- Voice recordings are decoded in memory and never written to disk.
- Live chat history is held in memory only. Long-term recall goes through the memory store, which the user can delete.
- The system prompt no longer describes the assistant as a professional therapist.
- The page no longer loads Tailwind or Google Fonts from third-party CDNs.
- Recordings are uploaded with their real container type (WebM or MP4) and decoded with PyAV.

### Removed

- The keyword emotion override. It ran before the acoustic model and matched substrings ("follow" matched `low`), and no measurement supported it.
- Conversation logs, voice recordings, the old vector store, `__pycache__` and `.DS_Store` files from version control. They remain in earlier commits until history is rewritten.
- `app.py`, `brain.py`, `train_emotion.py`, `transcribe.py`, replaced by the `soulsync/` package.
- The old `README_TRAINING.md` content, which described scripts that no longer existed.

## v1

Flask prototype: Whisper transcription, keyword and Wav2Vec2 emotion detection, Ollama `llama3` replies, macOS `say` speech output. Emotion model trained on 1,692 RAVDESS clips with a file-level split.
