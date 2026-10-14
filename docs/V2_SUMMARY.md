# SoulSync v2 summary

Written 2026-10-14. Numbers quoted here are copied from [RESULTS.md](../RESULTS.md), which is generated from `eval/results/`. If the two ever disagree, RESULTS.md is right.

## What changed

- **Backend:** Flask replaced by a FastAPI package (`soulsync/`) with the same streaming contract, Pydantic models and `/health`.
- **Emotion model:** retrained on four corpora with a speaker-independent split; exported to ONNX and quantised to INT8.
- **Emotion2Vec:** label mapping fixed, then compared against Wav2Vec2 on validation and enabled in local mode.
- **Memory:** ChromaDB memory rebuilt with per-user filtering, current-session exclusion, a tuned distance threshold and a delete endpoint.
- **Safety:** crisis-language layer that skips the LLM and suggests helplines; disclaimer and data-flow notice in the UI.
- **Identity:** signed one-year cookie per browser; a new session on each page load.
- **Speech output:** browser Web Speech API by default; macOS `say` optional.
- **Modes:** local (Ollama, local Whisper) and cloud (Groq, ONNX, no PyTorch) from one configuration module.
- **Evaluation:** scripts for emotion, ensemble, ONNX, memory and latency; `RESULTS.md`, the model card, the README table and the resume bullets are generated from their JSON output.
- **Weights published:** https://huggingface.co/npr1138/soulsync-emotion-wav2vec2 (PyTorch + INT8 ONNX, CC BY-NC-SA 4.0).
- **Record:** `training/SoulSync_training_and_eval.ipynb` holds every epoch, metric and confusion matrix with saved outputs.
- **Knowledge base:** ten draft coping-skill notes in a separate collection, threshold tuned by `eval/knowledge_eval.py`, off by default until you review the text.
- **Rate limiting:** 12 recordings per user per minute by default.
- **Reply quality:** automated LLM-as-judge plus keyword rule checks over 20 scripted conversations. The first run exposed a missed indirect crisis phrasing and missing medication referrals, both now fixed.
- **Licences:** checked against each corpus's source page. TESS is CC BY-NC 4.0 (earlier docs said BY-NC-ND). SAVEE forbids redistribution, so the Kaggle copy is itself a redistribution.
- **Tests and CI:** 123 pytest cases using fakes; a GitHub Actions workflow.

## Headline results

| | |
|---|---|
| v2 emotion model, 877 test clips, 14 unseen speakers | 71.2% accuracy, 71.1% macro-F1 |
| v1 recipe reproduced, same test set | 32.4% accuracy |
| Trained without TESS, tested on TESS | 60.6% accuracy; "neutral" recall collapses to under 1% |
| Emotion2Vec alone / ensemble | 89.3% / 89.5% accuracy, probably optimistic (see below) |
| ONNX INT8 | 122.1 MB against 378.3 MB, macro-F1 within half a point, p50 65.5 ms on CPU |
| Memory, held-out half | Hit@3 81.0%, false recall 0%, 0 cross-user leaks |
| Local latency, upload to first reply token | p50 0.65 s, p95 0.93 s |

## What was fixed from AUDIT.md

| Audit item | Status |
|---|---|
| 0.1-0.3 Dataset and weights missing | Dataset re-downloaded from Kaggle; model retrained. Original v1 weights never recovered |
| 0.5 Ambiguous Qwen tag | `qwen2.5:7b-instruct` |
| 1.1 Emotion2Vec scores dropped | Fixed, with a regression test; confirmed by running the model |
| 1.2 Only 1,692 clips used | 7,635 training clips from four corpora |
| 1.3 Speaker leakage, no test set | Speaker-disjoint train/val/test; overlap checks are zero |
| 1.4 Memory removed; cross-user query | Rebuilt with user filter; leak test in the eval and in pytest |
| 1.5 No persistent identity | Signed cookie |
| 1.6 macOS-only speech; argument injection | Browser speech by default; `say` reads text from stdin |
| 1.7 Missing and unpinned dependencies; stale training README | Four pinned requirements files; README_TRAINING rewritten |
| 1.8 Unused `emotion` field | Removed |
| 1.9 `llama3` hardcoded | `LLM_MODEL` |
| 1.10 Silent failures | Typed errors, logging, "Unavailable" shown in the UI |
| 1.11 Unbounded history | Last `HISTORY_MAX_TURNS` exchanges |
| 1.12 Personal data in git | Untracked and ignored. **Still present in earlier commits** |
| 1.13 No tests, eval, Docker, CI, licence | Tests, eval and CI added. MIT licence for code, CC BY-NC-SA 4.0 for weights. Dockerfile written but not built |
| 2.1 Keyword override | Removed |
| 2.2 No crisis handling | Safety layer with tests |
| 2.3 Train/serve mismatch | One shared preprocessing function |
| 2.4-2.8 Streaming and parsing bugs, LLM errors | Fixed; covered by API tests |
| 2.11 Shared `None` session | Requests without a valid cookie are rejected |
| 2.12 Recordings kept on disk | Decoded in memory, never saved |
| 2.13 No upload limit or rate limiting | `MAX_UPLOAD_MB` and `RATE_LIMIT_PER_MINUTE` (in memory, single process) |
| 2.14 Third-party CDNs | Removed |
| 2.15 No disclaimer | Added |
| 2.19 Emotion2Vec variant unclear | Still unclear: 164 M parameters, which matches neither size on the public model card |

## What was not done, and why

- **Docker build and memory measurement (Phase 7).** Docker is not installed and you asked to defer it. The Dockerfile and compose file are untested. `eval/results/cloud_memory.json` does not exist.
- **Cloud mode from a real server.** Groq was tested live from the laptop (24 clips, `latency_cloud.json`); `qwen/qwen3.8-27b` returned clean replies with no reasoning text. Not yet run from a deployed container.
- **Groq free-tier limits.** Their docs page lists Developer-plan limits only.
- **Original v1 baseline.** The original weights were unavailable, so the baseline is a reproduction. It reaches 61.9% on its own leaky validation split where your notebook reported 82.9%, so it is weaker than the real v1 model and the before/after gap is overstated.
- **`wav2vec2-base` versus `wav2vec2-base-960h` comparison. Attempted and failed.** With the main recipe the 960h model's loss became NaN within the first 100 steps. I stopped it after 4.1 minutes rather than retune it, because a different learning rate would no longer be a like-for-like comparison. Recorded in `eval/results/training_base960h.json`.
- **Git history purge.** Not done; it needs a force-push, which is your decision. Conversation logs and recordings remain recoverable from earlier commits on GitHub.
- **Knowledge base switched on.** The notes are a draft in my own words and need your review before `KNOWLEDGE_ENABLED=true`.
- **Memories in replies.** Retrieval injects the right memory, but the 7B model mentions it in only 1 of 4 judged cases. Not fixed.
- **A trustworthy reply-quality score.** The judge is the same model as the reply writer and gives near-ceiling scores; only the rule checks are informative.
- **Browser check.** The server was tested with real models through `curl`. The microphone, live text rendering and browser speech have not been tried in a browser.
- **Git history.** Purged and force-pushed. GitHub still serves the old commits by commit ID until GitHub Support removes them.

## Things to know before quoting results

- All emotion numbers are on acted English speech.
- Validation macro-F1 was 81.8% and test is 71.1%. With 14 speakers per held-out set, results move a lot between speaker groups.
- Emotion2Vec's model card says it was fine-tuned on EmoBox academic emotion data, which probably overlaps these corpora. Its score is not a clean speaker-independent result, and it supplies nearly all of the ensemble's accuracy.
- INT8 was 0.4 points better than FP32 on macro-F1. That is noise, not an improvement.
- ONNX FP32 was slower than PyTorch on this CPU (101.6 ms against 70.2 ms); only the INT8 model is faster.
- `openai-whisper` was faster than `faster-whisper` here (about 152 ms against 297 ms p50 on short clips), so it is the local default.

## Training time

| Run | Minutes |
|---|---|
| Timing benchmark | 0.95 |
| v1 recipe reproduction, 10 epochs | 10.8 |
| Main model, 12 epochs (early-stopped, best at epoch 9) | 96.3 |
| Leave-TESS-out, 8 epochs | 47.8 |
| `wav2vec2-base-960h` comparison, failed (NaN), plus a 30-step diagnostic | 4.1 |
| **Total** | **160.0 (2.67 h of the 4.5 h budget)** |

The benchmark projected 6.8 minutes per epoch for the main run; it took 8.0.
