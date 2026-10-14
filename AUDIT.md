# SoulSync v1 audit

Audited at commit `e32c33f` (branch `main`), 2026-10-13. Every item was checked against the code, the notebook's saved outputs, or git history. Items marked **(not verifiable yet)** depend on files that are not on this machine (see section 0).

Severity: **P0** = wrong result, privacy/safety problem, or app cannot run. **P1** = blocks deployment or reproducibility. **P2** = quality or hygiene.

## 0. Blockers found before any code was read

| # | Finding | Evidence |
|---|---|---|
| 0.1 | The local project copy `~/Desktop/SoulSync` does not exist on this Mac. | `ls ~/Desktop/SoulSync` fails. Spotlight and `find` locate no `Emotions` folder with audio, no RAVDESS/CREMA-D/TESS filenames, and no `model.safetensors` outside caches. |
| 0.2 | The dataset is not in git. `custom_model/Emotions/` contains only a `.DS_Store`. | `git ls-files custom_model/` |
| 0.3 | The fine-tuned weights are not in git. `custom_model/final_emotion_model/` has `config.json` and `preprocessor_config.json` only; `model.safetensors` is git-ignored (`.gitignore:7`). | `git ls-files custom_model/` |
| 0.4 | Docker is not installed. Phase 7 needs it to build the image and measure memory. | `command -v docker` returns nothing |
| 0.5 | Two Qwen 2.5 text models are installed, so "the Qwen 2.5 tag" is ambiguous: `qwen2.5:7b-instruct` (4.7 GB) and `qwen2.5:3b-instruct` (1.9 GB). `qwen2.5vl:7b` is a vision model and not a candidate. | `ollama list` |

Consequence: a fresh clone of this repo cannot run (no weights) and cannot retrain (no data). The v1 baseline number and all of Phase 5 are blocked until the dataset and weights are supplied.

## 1. The issues listed in the brief

### 1.1 Emotion2Vec contributes nothing to the "60/40 ensemble" (P0, confirmed)

- `models/emotion2vec/tokens.txt` holds bilingual labels: `生气/angry`, `厌恶/disgusted`, `恐惧/fearful`, `开心/happy`, `中立/neutral`, `其他/other`, `难过/sad`, `吃惊/surprised`, `<unk>`.
- `brain.py:131-134` tests `lbl.lower() in {"happy","sad","angry","neutral","anxious"}`. `"生气/angry".lower()` is never in that set, so `e2v_probs` stays all zeros.
- `brain.py:137` then computes `0.6 * w2v_probs + 0`. The argmax equals Wav2Vec2's argmax. The 1.9 GB model is loaded and run on every request for no effect.
- Even with the prefix stripped, `fearful` would not map to `anxious`.
- The claim rests on the label file and the code. I could not run the model to observe the zeros directly because `model.pt` has not been downloaded. Phase 5 adds a test that does.

### 1.2 The notebook trained on 1,692 clips (P0, confirmed from saved output)

- `custom_model/train.ipynb` cell 2 splits each filename on `-` and reads `parts[2]` as a RAVDESS emotion code. Any other naming scheme is skipped with no warning.
- Saved output: `Found 1692 files matching your 5 emotions`, 1,353 train / 339 validation.
- Class counts in the train split are 301/301/301/300/150. That matches RAVDESS speech plus RAVDESS song for 24 actors (376 per emotion, 188 neutral). So the model probably also learned from sung clips. This is my inference from the counts, to be confirmed in the Phase 5 inventory.
- README.md:342-350 lists 12,798 files across seven class folders (and says "12805" total). **(not verifiable yet)**

### 1.3 Speaker leakage and no test set (P0, confirmed)

- Cell 2: `train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])` splits by file. The same 24 actors, speaking the same two sentences, are on both sides.
- The best checkpoint was selected on validation accuracy (cell 5, `metric_for_best_model="accuracy"`) and the reported figure is that same validation set (cell 7). There is no held-out test set.
- The reported number is 0.829 accuracy on 339 clips (macro-F1 0.82). It is not a speaker-independent figure and should not be quoted as one.

### 1.4 RAG memory was added, then removed (P0 privacy bug in the old version, confirmed with one correction)

- ChromaDB memory is present in three commits, `51cc8e7`, `9310137` and `deab19a`, and gone from `934946d` onward. The brief says it was deleted in the next commit; it survived two more.
- `51cc8e7:brain.py:30` queries with `self.collection.query(query_texts=[current_input], n_results=3)`. There is no `where` filter, no user ID and no distance threshold, so every user's text could be retrieved into anyone's prompt.
- `51cc8e7:brain.py:31-32` wraps that in a bare `except: return ""`.
- `data/vector_vault/` is still tracked (5 files). `chroma.sqlite3` holds 8 stored `User: … | AI: …` documents, which is real conversation text.

### 1.5 No persistent user identity (P1, confirmed)

- `app.py:11` sets `app.secret_key = os.urandom(24)`, so cookies are invalid after every restart.
- `app.py:22` assigns a new `session['id']` on every `GET /`.
- `app.py:23-29` writes a new history file on every page load. 23 of the 57 tracked history files contain no user turn.

### 1.6 macOS-only speech output (P1, confirmed)

- `app.py:61-67` runs `say -v Samantha -r 150 -o <file> <text>`.
- The text is passed as a positional argument. A reply starting with `-` would be parsed as a `say` option. There is no length limit.
- Output files in `static/audio/` are never deleted and are publicly served.

### 1.7 Missing and stale dependencies (P1, confirmed)

- `brain.py:10` imports `transformers`; `requirements.txt` does not list it. Also unlisted: `datasets`, `scikit-learn`, `matplotlib`, `seaborn` (notebook) and `ffmpeg` (system dependency, see 2.6).
- `requirements.txt` pins nothing. `pandas` and `torchvision` are listed but unused at runtime.
- `README_TRAINING.md` documents `generate_synthetic_data.py`, `train_bridge.py`, `therapist_bot.py` and `prep_data.py`. None exist. It says nothing about the Wav2Vec2 notebook that produced the deployed model.
- `train_emotion.py` trains nothing; it is an Emotion2Vec inference wrapper.

### 1.8 Unused `emotion` field (P2, confirmed)

`templates/index.html:154` sends `{text, emotion}`; `app.py:52-53` reads `text` only.

### 1.9 LLM hardcoded to `llama3` (P0, confirmed)

`brain.py:33` sets `self.model_name = "llama3"`. That model is not installed, so `ollama.chat` at `brain.py:167` fails on the first real message.

### 1.10 Silent failures (P1, confirmed)

- `brain.py:141-143`: any exception in the neural path prints and returns `"Neutral"`. A missing model, a decode error and a true neutral prediction are indistinguishable to the user and in the logs.
- `brain.py:106-107`: energy-check errors are printed and ignored.
- `templates/index.html:169`: `catch (e) { statusText.innerText = "Error"; }` discards the error.
- There is no bare `except:` in HEAD; the ones in the brief's sense are these broad `except Exception` blocks. Logging is `print` only.

### 1.11 Unbounded history (P1, confirmed)

`brain.py:163-167` loads the whole history file and sends all of it to the LLM each turn. Nothing truncates it.

### 1.12 Personal data in git (P0, confirmed, count is exact)

- 92 tracked files under `data/history/` (58: 57 JSON plus `.DS_Store`) and `static/audio/` (34). 34 of the history files contain real user turns.
- `data/raw_audio/` is not tracked in HEAD, but 7 voice recordings exist in earlier commits and are recoverable from history.
- Also tracked: `data/vector_vault/` (5 files, see 1.4), `static/speech.aiff`, `data/test_embedding.npy`, 5 `__pycache__/*.pyc`, 7 `.DS_Store`.
- Root cause: `.gitignore:6` reads `history/custom_model/Emotions.zip`. Two rules were fused onto one line, so neither `history/` nor `custom_model/Emotions.zip` is ignored.
- Secrets: I scanned all history for API keys. Deepgram and Inworld keys were always read from environment variables and `.env` was never committed. No leaked credential found.

### 1.13 No tests, eval scripts, Docker, CI or licence (P1, confirmed)

None of `tests/`, `eval/`, `Dockerfile`, `.github/workflows/` or `LICENSE` exist.

## 2. Additional issues found

### Correctness

| # | Sev | Issue | Reference |
|---|---|---|---|
| 2.1 | P0 | Keyword matching runs before the neural model and uses substring tests, so it overrides the model on most real sentences and misfires: `low` matches "follow", `mad` matches "made", `cross` matches "across", `content` matches the noun, `lost`, `blue`, `tired`, `sorry`. Negation is ignored ("I'm not sad" returns Sad). Dictionary order makes Sad win ties. The neural model that the README headlines is a fallback. | `brain.py:110-113`, `brain.py:43-91` |
| 2.2 | P0 | No crisis handling. The old `check_safety` (`51cc8e7:brain.py:24-26`) was removed, so self-harm statements go straight to the LLM. | `brain.py:157-177` |
| 2.3 | P1 | Train/serve mismatch. Training padded or cut every clip to 3 s (`max_length=48000`); inference feeds the full recording with no crop. | notebook cell 3; `brain.py:122-123` |
| 2.4 | P1 | The reply is not shown as it streams. The frontend buffers the whole stream, then waits for TTS, and only reveals the text when audio starts. | `index.html:133-148`, `161-165` |
| 2.5 | P1 | Stream parsing is fragile. A `|` in the transcript shifts the fields. The header is assumed to arrive in one chunk. `decoder.decode(value)` lacks `{stream:true}`, so multi-byte characters split across chunks are corrupted. | `index.html:136-146`, `app.py:44` |
| 2.6 | P1 | The upload is labelled `audio/wav` and saved as `.wav`, but `MediaRecorder` produces WebM/Opus (or MP4 on Safari). Decoding only works through librosa's audioread fallback, which needs ffmpeg. That dependency is undocumented. | `index.html:105,116`, `app.py:37` |
| 2.7 | P1 | Action-stripping regex is applied to each streamed chunk separately, so `*sighs*` split across chunks is not removed. | `brain.py:172` |
| 2.8 | P1 | If Ollama errors, the generator raises mid-stream and the user sees a truncated reply with no message. | `brain.py:167-174` |
| 2.9 | P2 | Audio is decoded twice per request (energy check, then model). Emotion2Vec is run with `extract_embedding=True` and the embedding is discarded. | `brain.py:101,122,129`; `train_emotion.py:34` |
| 2.10 | P2 | The silence fallback says "How are you feeling this wonderful day?", a poor default for someone in distress. | `brain.py:160` |

### Privacy and security

| # | Sev | Issue | Reference |
|---|---|---|---|
| 2.11 | P0 | If the session cookie is missing, `session.get('id')` is `None` and every such user writes to `data/raw_audio/None_input.wav` and `data/history/None.json`. Users would share a history. | `app.py:36-37`, `brain.py:146` |
| 2.12 | P1 | Raw voice recordings are kept on disk indefinitely, one per session, with no deletion path. Same for history JSON and TTS output. | `app.py:37`, `brain.py:152-155` |
| 2.13 | P1 | No upload size limit, no content-type check, no rate limiting. | `app.py:32-38` |
| 2.14 | P2 | A "local" app loads Tailwind and Google Fonts from third-party CDNs on every page view. | `index.html:7,9` |
| 2.15 | P1 | No disclaimer. The system prompt calls the model "a professional and grounded therapist" and the page title says "Neural Voice Therapy". | `brain.py:35`, `index.html:6` |

### Evaluation and reproducibility

| # | Sev | Issue | Reference |
|---|---|---|---|
| 2.16 | P1 | The notebook reports accuracy only during training; no seed is set for the model head or data order; paths are absolute (`/Users/neerajprao/Desktop/SoulSync/...`). | notebook cells 2, 5, 6 |
| 2.17 | P1 | The "83%" is one run on 339 clips. A 95% interval at that size is roughly ±4 points, before accounting for leakage. | notebook cell 7 |
| 2.18 | P1 | RAVDESS uses two fixed sentences. A model can score well on it without generalising to free speech. | dataset property |
| 2.19 | P1 | The Emotion2Vec variant is unclear. `configuration.json` says `iic/emotion2vec_base`, but a 9-label `tokens.txt` and a 1.9 GB checkpoint suggest a fine-tuned "plus" variant. To be checked when the weights are available; it affects the model card and licence. | `models/emotion2vec/` |
| 2.20 | P2 | README figures do not match the repo: "55 history files" (57 tracked), "62 audio files" (34 tracked), "over 71k files". | `README.md:363,398,487` |

### Hygiene

| # | Sev | Issue | Reference |
|---|---|---|---|
| 2.21 | P2 | `transcribe.py` duplicates the Whisper code in `brain.py` and is unused by the app. `train_emotion.py` prints at import time and imports `librosa` without using it. | `transcribe.py`, `train_emotion.py:2,9-10` |
| 2.22 | P2 | A 1.9 GB file sits in GitHub LFS. The free LFS quota is small, so clones by others may fail once bandwidth runs out. | `.gitattributes` |
| 2.23 | P2 | Port 5000 collides with macOS AirPlay Receiver on recent macOS versions. | `app.py:79` |
| 2.24 | P2 | `how_to_open.txt` and the README refer to `venv/` with Python 3.14, on which `funasr` and `onnxruntime` are not reliably installable. | `how_to_open.txt` |

## 3. What could not be verified

1. Anything about the dataset's real contents: file counts, corpora, speakers, licences (section 0).
2. The v1 model's behaviour, including a live demonstration of 1.1 and the speaker-independent baseline.
3. Whether the app ran end to end in v1. It cannot run here: no weights, no `llama3`, no Emotion2Vec checkpoint.
