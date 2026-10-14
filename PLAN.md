# SoulSync v2 plan

Workspace: `~/Desktop/project/` with venv `.venv` (Python 3.11.16) and the repo in `Ai_therapist/` on branch `v2`. Written before the work started; see docs/V2_SUMMARY.md for what happened.

## Decisions needed before work continues

1. **Where are the dataset and the fine-tuned weights?** `~/Desktop/SoulSync` does not exist and neither is in git. Phase 5 and the v1 baseline cannot start without them. Phases 1 to 4 can.
2. **Which Qwen 2.5 tag?** `qwen2.5:7b-instruct` or `qwen2.5:3b-instruct`. I recommend 7b as the default for reply quality and will measure latency for whichever is chosen in Phase 8.
3. **Emotion2Vec `model.pt` (1.9 GB):** do you have it locally, or should I fetch it from LFS when Phase 5 needs it?
4. **Docker:** not installed. Phase 7 needs it. Installing Docker Desktop, OrbStack or Colima is a system-wide install, so it is your call.

## Phases

| Phase | Work | Output | Needs from you |
|---|---|---|---|
| 1. Hygiene and structure | Untrack personal data, caches and junk; fix `.gitignore`. Move code into `soulsync/` (`config`, `llm`, `stt`, `emotion`, `memory`, `safety`, `tts`, `api`) with `training/`, `eval/`, `tests/`, `scripts/`, `docs/`. Env-driven `config.py`, `.env.example`, logging, no broad silent excepts. | Clean tree, app still starts | Whether to purge history with `git filter-repo` (needs a force-push) |
| 2. FastAPI | Port endpoints with the same `METADATA|transcript|emotion|` first chunk. Pydantic models, `/health`, 1-year HTTP-only `SameSite=Lax` user cookie, new `session_id` per page load, browser `speechSynthesis` by default with `say` optional, last-N-turn history. Fix stream parsing and show text as it streams. | Working app on FastAPI | none |
| 3. Memory | `soulsync/memory.py` on ChromaDB (cosine, MiniLM ONNX). Store with `user_id`, `session_id`, emotion, timestamp. Retrieve with `user_id ==` and `session_id !=`, distance threshold, ephemeral system message. `DELETE /memory` and a UI button. | Memory with isolation tests | Whether to add the coping-skills knowledge base, and its sources |
| 4. Safety | Keyword and regex crisis detector with normalisation; fixed helpline reply that bypasses the LLM; UI disclaimer; unit tests. | `soulsync/safety.py` and tests | Confirm helpline numbers (Tele-MANAS 14416) |
| 5. Emotion model | Inventory, speaker IDs, `docs/DATASET.md` (checkpoint). 5-class mapping, speaker-disjoint 80/10/10 split saved as CSV. Baseline of the v1 model on the new test set. Fine-tune, test once. Fix Emotion2Vec mapping, sweep ensemble weight on validation, report once on test. Model card. | `eval/results/emotion_*.json`, confusion matrix, `docs/MODEL_CARD.md` | Dataset and weights; approval of the dataset table; approval before any Hub upload |
| 6. ONNX | Export, dynamic INT8 quantisation, benchmark size, CPU latency p50/p95, accuracy and macro-F1. ONNX path runs without PyTorch. | `eval/results/onnx_benchmark.json` | none |
| 7. Local and cloud modes | Local: Ollama, Whisper (openai-whisper vs faster-whisper, measured), torch or ONNX emotion. Cloud: Groq LLM and STT, ONNX INT8, ChromaDB, browser TTS, `requirements-cloud.txt`, Dockerfile, compose, measured image size and peak memory. Groq model names and limits taken from their docs. | `eval/results/cloud_memory.json` | Docker; a Groq key for cloud measurements |
| 8. Evaluation | Scripts under `eval/` writing JSON; a generator that builds `RESULTS.md` from the JSON. Memory retrieval set (3+ users, 40+ follow-ups, 15+ unrelated), threshold sweep on half and report on the other half. End-to-end latency over 20+ clips. | `RESULTS.md` | Whether to run the optional LLM-as-judge evaluation |
| 9. Tests and CI | pytest for memory, safety, config, label mapping and API with fakes. GitHub Actions on push, CPU only, no downloads. | `tests/`, workflow | none |
| 10. Docs | README with Mermaid diagram and results from `RESULTS.md`, new `README_TRAINING.md`, `CHANGELOG.md`. | Docs | Which licence |
| 11. Final report | Full rerun, `docs/V2_SUMMARY.md`, `docs/RESUME_BULLETS.md` with a proving file under each bullet. | Final checkpoint | Approval before push or PR |

## Training time projection

This is a projection from the old notebook's saved output, not a measurement. It will be replaced by a 50 to 100 step benchmark on this machine before any long run, and I will report the benchmark and the chosen epoch count before starting.

- Notebook: 13,530 training samples in 887.6 s on MPS, including per-epoch evaluation, so about 15.2 clips/s (batch 4, 3 s clips, encoder CNN frozen).
- Expected 5-class pool, from the README's folder counts: 2,167 + 2,167 + 2,167 + 1,795 + 2,047 = 10,343 clips. An 80% train split is about 8,270 clips.
- That gives about 9 minutes of training per epoch plus about 1 minute of validation.

| Item | Projected | Counts toward 4.5 h |
|---|---|---|
| Timing benchmark | 5 min | yes |
| Main fine-tune, up to 12 epochs, early stopping on validation macro-F1 with patience 3 | 2 h, hard cap 3 h | yes |
| Cross-corpus run (leave one corpus out), about 5 epochs | 50 min | yes, only if at least 1 h remains |
| Reserve for a resume after a crash | 30 min | yes |
| Projected total | about 3 h 25 min | of 4.5 h |

Feature caching, baseline evaluation, Emotion2Vec inference, the ensemble sweep and ONNX benchmarks are inference only. I will not count them as training time but will log their durations.

The `wav2vec2-base` versus `wav2vec2-base-960h` comparison does not fit alongside the cross-corpus run. I plan to train `facebook/wav2vec2-base` only, and note the comparison as not done, unless you prefer it over the cross-corpus test. `wav2vec2-base` is a 360 MB download; `wav2vec2-base-960h` is already cached.

## Risks

1. **Missing data and weights.** Without the v1 weights there is no honest "before" number. If they are gone for good, the fallback is to rerun the old notebook recipe on the new split and label that result clearly as a reproduction, not the original model. That would cost about 15 minutes of training budget.
2. **TESS has two speakers and roughly a fifth of the clips.** A speaker-disjoint split cannot be 80/10/10 and balanced at once if a TESS speaker lands in validation or test. I will show options at the dataset checkpoint.
3. **The new number will likely be lower than 83%.** Speaker-independent accuracy on mixed acted corpora is harder than the leaky RAVDESS split. That is the truthful figure and the one to quote.
4. **Dataset licences.** If the corpora are what the filenames suggest: RAVDESS is CC BY-NC-SA 4.0, TESS is CC BY-NC-ND 4.0, CREMA-D is ODbL, and SAVEE is research-use by registration. Non-commercial terms may restrict publishing trained weights on the Hub. I will set out the position in `docs/DATASET.md` and ask before uploading anything.
5. **Cloud memory limit.** ChromaDB's MiniLM ONNX embedder, the INT8 emotion model and audio decoding together may exceed 512 MB. I will measure and report what uses the memory instead of assuming it fits.
6. **MPS quirks.** Some ops fall back to CPU; funasr on MPS may not work and Emotion2Vec may have to run on CPU, which is slower but does not affect results.
7. **History purge.** Removing personal data from history rewrites every commit hash and needs a force-push. Until then the recordings and logs stay recoverable from GitHub even after Phase 1.
8. **Keyword override.** v1 lets keywords overrule the acoustic model. I plan to make the acoustic model the source of the emotion label and keep keywords off by default, since no measurement supports them. Tell me if you want them kept.
