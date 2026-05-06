# CStore — Web Console

A single project that contains both the UI and the backend for driving the
CStore text-to-Csound generator locally.

- **UI** — Next.js 16 App Router + React 19 + Tailwind CSS v4 (editorial / studio
  console aesthetic).
- **Backend** — a small Flask sidecar in `server/` that loads the PyTorch model,
  shells out to the `csound` CLI to render audio, and writes outputs to
  `generated/`. The Next.js dev server proxies `/api/*` and `/generated/*` to
  this sidecar via [`next.config.ts`](./next.config.ts).

> **Why two processes?** The model is a Hugging Face GPT-2 checkpoint on
> `.safetensors` + a BPE tokenizer loaded through Python, and rendering
> requires running the `csound` command-line tool. Running that from Node
> honestly isn't feasible, so the console keeps a Python sidecar running
> locally. There is no Flask HTML anymore; Flask only exposes JSON endpoints
> and the generated files.

## Layout

```
webapp-next/
├── app/                         # Next.js UI
│   ├── _components/
│   │   ├── Console.tsx          # main studio layout (library, transport, source)
│   │   ├── CsoundTerminal.tsx   # live Csound / sidecar log viewer
│   │   ├── EditDialog.tsx       # LLM edit modal (6 providers)
│   │   ├── ManualEditDialog.tsx # hand-edit + re-render modal
│   │   ├── ModelPicker.tsx      # checkpoint switcher
│   │   └── Spectrum.tsx         # WebAudio FFT monitor
│   ├── _lib/api.ts              # typed fetchers
│   └── globals.css              # Tailwind v4 @theme tokens
├── server/
│   └── app.py                   # Flask API (generation, starter batches, Csound render, edit, favorite)
├── generated/                   # rendered .csd + .wav + meta.json per run (gitignored)
├── next.config.ts               # proxies /api/* and /generated/* to 127.0.0.1:5000
└── package.json                 # `npm run dev` and `npm run server`
```

## Requirements

- Node.js 20+
- Python 3.11+ with the deps in the repo root: `pip install -r ../requirements.txt`
- Use the same Python interpreter/venv for install and runtime (so `npm run server` sees installed deps)
- [Csound](https://csound.com/) 6.18+ on your `PATH` (the `csound` binary)
- At least one checkpoint in `../model/checkpoints/` (default: `Cstore_V1.0.2/best`)
- Checkpoints are not committed to git; download from [Releases](https://github.com/CharlieSL1/CStore/releases) if missing.
- `soundfile` may require system `libsndfile` (`brew install libsndfile` on macOS).

## Run it

You need **two terminals** — one for the backend, one for the UI. From this
folder:

```bash
# terminal 1 — backend (Flask on :5000)
npm run server

# terminal 2 — UI (Next.js on :3000)
npm install      # first time only
npm run dev
```

Open http://localhost:3000.

`npm run server` tries `python3 server/app.py` first, then falls back to
`python server/app.py`. You can also run those commands directly if you prefer
a Python venv.

Recommended venv flow:

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd webapp-next
npm run server
```

If backend startup fails, check:

```bash
# from webapp-next/
python3 --version || python --version
ls ../model/checkpoints
csound --version
```

To point the UI at a backend on a different host/port, set
`CSTORE_BACKEND_URL` before `npm run dev` — see `next.config.ts`.

### Environment variables

Required for basic local run:

- none (defaults work when backend is at `http://127.0.0.1:5000`).

Optional:

- `CSTORE_BACKEND_URL`: override Next.js proxy target (default `http://127.0.0.1:5000`).
- `CSTORE_OLLAMA_URL`: Qwen/Ollama endpoint (default `http://127.0.0.1:11434`).
- `CSTORE_POLLINATIONS_URL`: Pollinations endpoint (default `https://text.pollinations.ai`).
- `CSTORE_OPENROUTER_URL`: OpenRouter endpoint (default `https://openrouter.ai/api/v1`).
- `CSTORE_COST_*`: per-provider token price overrides used for cost estimation.

## Keyboard shortcuts

| Key       | Action                                                        |
|-----------|---------------------------------------------------------------|
| `G`       | Generate one render                                           |
| `S`       | Generate starter ideas batch (outputs / max tok controls)    |
| `Space`   | Play / pause render                                           |
| `C`       | Copy CSD source                                               |
| `E`       | Open the *Edit with LLM* dialog for the active run            |
| `R`       | Open the *Edit manually* window (hand-edit + re-render)       |
| `☆`       | Click a library row's star to pin it to **Favourites**        |
| `⌘/Ctrl ↵`| Submit the active dialog                                      |
| `Esc`     | Close the active dialog                                       |

## Library & Favourites

The left rail splits into **Favourites** and **Recent**. Click the star on
any row to pin a timbre you like — pinned runs move into the Favourites
section and stay there even after the 50-row recent cap rolls them out of
the main list. The flag is persisted to that run's `meta.json`, so it
survives restarts.

The star is backed by `POST /api/favorite` (`{run_id, favorite?}`);
omitting the `favorite` field toggles the current value.

## Starter batches

The monitor panel has dedicated actions for:

- **Generate one render** (`G`) — one fresh `.csd` + `.wav`.
- **Generate starter ideas** (`S`) — multiple short related starts.
- **Generate children** — from the selected mother/parent run.

Use **outputs** to choose how many renders to make
(`1`-`12`) and **max tok** to request a larger generation budget than the old
400-token default (`100`-`500`). The backend still clamps the actual token
budget to the model context window, so a request above the checkpoint's limit
will not overflow the GPT-2 context.

For `arpeggio`/`chord`, note selection supports:

- **auto 2oct** — random distinct notes inside a 2-octave span.
- **manual** — user-provided low/high MIDI bounds.

Every starter is rewritten to a short score audition and tagged in
`meta.json` as `kind: "starter"`. Modes are:

- `single` — one note across the selected duration.
- `arpeggio` — four short notes inside the same 1.5-second window.
- `chord` — three simultaneous notes lasting 1.5 seconds.

Starter/child outputs are stored in one persistent panel as batch tabs. Each
new generate action appends a new tab (`Starter #...`, `Classic Child #...`,
`LLM Child #...`) instead of replacing the current view. You can switch tabs,
close a single batch, or clear all batches. Child cards inside the active tab
still load that run's `.csd` + `.wav` into the existing player/source inspector.

Tag semantics in the UI are strict:

- **starter** = generated starter variation.
- **child** = generated from a selected mother/parent run (shows `mother #...` lineage).

Children generation supports two modes:

- **classic** — deterministic source-derived score rewrites (`/api/generate-children`).
- **llm variants** — provider/model-based automatic variation prompts (`/api/generate-children-llm`), no manual prompt required.

`LLM VAR` drives tiered behavior:

- **low** (`0.00-0.33`) — subtle changes close to the mother.
- **medium** (`0.34-0.66`) — moderate timbre/register/articulation shifts.
- **high** (`0.67-1.00`) — aggressive style-switch directives with explicit
  anti-similarity constraints and stronger fallback divergence.

## Editing a run

Two editors share the run-card footer:

### Edit manually (`R`)

Opens a monospace editor seeded from the current `.csd`. Edit by hand, press
`⌘/Ctrl ↵` to send the text back through `POST /api/render`, which writes
it as a fresh run and invokes the `csound` binary locally — no model, no
network. The result loads into the console if Csound produces audio; if the
source has an error the `.csd` is still saved and the Csound error surfaces
in the dialog. Manual runs are tagged **MANUAL** in the library.

### Edit with an external LLM (`E`)

Any existing run can be sent to one of six providers along with a
natural-language instruction (e.g. *"Add a plate reverb, lower the pitch by
an octave"*). The model returns a revised `.csd`, which the backend renders
with Csound and saves as a **new** run (the original is never overwritten).
LLM-edited runs are marked **EDIT** in the Library rail and show their
provenance (source run · provider · model · instruction) in the Run card.
When usage metadata is available from a provider, the run metadata also stores
token counts and an estimated cost payload.

| Provider               | Key required | Default models (user-editable)                                                           |
|------------------------|:------------:|------------------------------------------------------------------------------------------|
| **Qwen · local**       | ✗ (Ollama)   | `qwen3.6:35b-a3b-coding-mxfp8`, `qwen3.6`, `qwen3:14b`, `qwen3:8b`, `qwen2.5-coder:7b` |
| **Pollinations · free**| ✗            | `openai` → OpenAI's open-weight GPT-OSS-20B, served free via [Pollinations](https://github.com/pollinations/pollinations/blob/master/APIDOCS.md) at ~1 req/15 s |
| **OpenRouter · free**  | ✓            | `openrouter/free` (router), `qwen/qwen3-coder:free`, `openai/gpt-oss-20b:free`          |
| OpenAI                 | ✓            | `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.3-instant`                             |
| Anthropic              | ✓            | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`                      |
| Gemini                 | ✓            | `gemini-3.1-pro-preview`, `gemini-3-flash-preview`                                       |

The model field is free text, so newer IDs work without a code change.
Qwen reads `CSTORE_OLLAMA_URL` (default `http://127.0.0.1:11434`);
Pollinations reads `CSTORE_POLLINATIONS_URL` (default
`https://text.pollinations.ai`); OpenRouter reads `CSTORE_OPENROUTER_URL`
(default `https://openrouter.ai/api/v1`).

#### OpenRouter free setup (requires key)

1. Create an account at [OpenRouter](https://openrouter.ai/).
2. Generate an API key at [OpenRouter keys](https://openrouter.ai/settings/keys).
3. In the app's **Edit with LLM** dialog, choose **OpenRouter · free**, paste
   the key, and save it.
4. Keep the model as `openrouter/free` for auto-routed free inference, or
   type a pinned free variant such as `...:free`.

OpenRouter's free model pool and free-tier limits can change over time; check
their docs before production use.

### How the API key is handled

- Stored **server-side only** in `~/.cstore/keys.json` with file mode `0600`.
- The key never reaches the browser. The dialog only ever receives a mask
  (`sk-p…LAST`, first 4 + last 4). The "show" toggle on the input is for
  when you're typing a fresh key — the backend never echoes the full value
  back.
- `~/.cstore/` is outside this repo; it is not tracked by git.
- The key is **not encrypted at rest**. Without a user-supplied master
  password any "encryption" would just be obfuscation, so the real
  protection is the file mode + the fact that nothing ever sends it to the
  frontend. If you need stronger guarantees, manage the file yourself and
  make it a symlink to a location on an encrypted volume.
- To remove a key, click **remove** in the dialog or `DELETE /api/keys` with
  `{"provider": "openai"}`.

### Cost tracking for LLM edits

- Free providers (`qwen`, `pollinations`) are logged with `estimated_usd: 0`.
- Paid providers log token usage when returned by the upstream API.
- Estimated USD for paid providers is only computed when provider rates are
  configured through environment variables:
  - `CSTORE_COST_OPENAI_PROMPT_PER_1M`, `CSTORE_COST_OPENAI_COMPLETION_PER_1M`
  - `CSTORE_COST_ANTHROPIC_PROMPT_PER_1M`, `CSTORE_COST_ANTHROPIC_COMPLETION_PER_1M`
  - `CSTORE_COST_GEMINI_PROMPT_PER_1M`, `CSTORE_COST_GEMINI_COMPLETION_PER_1M`
  - `CSTORE_COST_OPENROUTER_PROMPT_PER_1M`, `CSTORE_COST_OPENROUTER_COMPLETION_PER_1M`
- If rates or usage breakdown are missing, cost is returned as `null` (never fabricated).

### Relevant API endpoints

| Method   | Path                         | Purpose                                                                     |
|----------|------------------------------|-----------------------------------------------------------------------------|
| `GET`    | `/api/keys`                  | Per-provider `{present, masked}`                                            |
| `POST`   | `/api/keys`                  | `{provider, key}` — store                                                   |
| `DELETE` | `/api/keys`                  | `{provider}` — remove                                                       |
| `GET`    | `/api/qwen/status`           | Ollama reachability + installed qwen tags                                   |
| `GET`    | `/api/pollinations/status`   | Pollinations reachability + anonymous-tier models                           |
| `GET`    | `/api/health`                | Sidecar availability probe used by the UI refresh path                      |
| `POST`   | `/api/generate-starters`     | `{count, seed?, checkpoint?, max_tokens?}` — make multiple 1.5s starter children |
| `POST`   | `/api/generate-children`     | `{source_run_id, count, note_duration?, note_mode?, note_range_mode?, note_low_midi?, note_high_midi?}` — derive classic children |
| `POST`   | `/api/generate-children-llm` | `{source_run_id, count, provider, model, think?, variation_temperature?}` — derive LLM children variants |
| `POST`   | `/api/edit`                  | `{run_id, provider, model, instruction, think?}` — LLM edit, returns a new run |
| `POST`   | `/api/render`                | `{csd, derived_from?}` — render hand-edited source, returns a new run       |
| `POST`   | `/api/favorite`              | `{run_id, favorite?}` — pin/unpin in the Library (toggles if `favorite` omitted) |

## Changing the model

The **Model** strip under the status bar has:

- a preset dropdown populated from `GET /api/models` (every folder under
  `../model/checkpoints/` that contains a `config.json`, grouped by family);
- a **path** input that accepts either a repo-relative label like
  `Cstore_V1.0.2/best` or any absolute path on disk. Press Enter or click
  **load** to apply.

The default checkpoint is `Cstore_V1.0.2/best`, set in `server/app.py`.

## API surface (for reference)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Health / endpoint index |
| `POST` | `/api/generate` | Draft `.csd` + render `.wav`. Body: `{ seed?, checkpoint?, max_tokens?, note_duration?, note_mode?, note_range_mode?, note_low_midi?, note_high_midi? }` |
| `POST` | `/api/generate-starters` | Draft multiple short starter children. Body: `{ count, seed?, checkpoint?, max_tokens?, note_duration?, note_mode?, note_range_mode?, note_low_midi?, note_high_midi? }` |
| `POST` | `/api/generate-children` | Derive classic children from a selected run. Body: `{ source_run_id, count, note_duration?, note_mode?, note_range_mode?, note_low_midi?, note_high_midi? }` |
| `POST` | `/api/generate-children-llm` | Derive LLM children variants. Body: `{ source_run_id, count, provider, model, think?, variation_temperature? }` |
| `GET`  | `/api/health` | Lightweight backend health probe |
| `GET`  | `/api/list` | List completed runs (+ pinned favourites) |
| `GET`  | `/api/console` | Live console ring buffer (polled by the web terminal) |
| `GET`  | `/api/models` | Discoverable checkpoints under `model/checkpoints/` |
| `GET`  | `/api/model` | Currently active checkpoint |
| `POST` | `/api/model` | `{ "path": "…" }` — swap the active checkpoint |
| `POST` | `/api/edit` | LLM edit (see above) |
| `POST` | `/api/render` | Manual-edit re-render (see above) |
| `POST` | `/api/favorite` | Toggle a run's favourite flag |
| `GET`  | `/generated/<run_id>/output.csd` | Source download |
| `GET`  | `/generated/<run_id>/output.wav` | Rendered audio |

## Global reverb (playback)

The monitor panel exposes a **reverb** wet-mix control (`0..1`) that applies at
playback time in the WebAudio graph. It is global for all loaded runs (base,
starter, and child), and does not rewrite the underlying `.csd`.

## Production build

```bash
npm run build && npm run start   # UI on :3000
npm run server                    # backend on :5000
```

The Flask sidecar is a development server; for a real deployment put it
behind gunicorn or similar and change `CSTORE_BACKEND_URL` accordingly.
