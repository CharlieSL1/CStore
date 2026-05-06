"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteKey,
  editRun,
  fetchPollinationsStatus,
  fetchQwenStatus,
  listKeys,
  saveKey,
  type EditResponse,
  type KeyProvider,
  type KeyStatusMap,
  type LlmProvider,
  type PollinationsStatus,
  type QwenStatus,
} from "../_lib/api";

type Props = {
  open: boolean;
  runId: string | null;
  /** Close the dialog without doing anything. */
  onClose: () => void;
  /** Called when an edit succeeds with the new run_id so the console can load it. */
  onEdited: (result: EditResponse) => void;
};

type ProviderDef = {
  id: LlmProvider;
  label: string;
  models: string[];
  /** True iff this provider is served by something running on the user's
   *  own machine (currently only Ollama). Suppresses the key UI. */
  local: boolean;
  /** True iff this provider is a free public endpoint that needs no key
   *  and no local install. Currently only `pollinations`. */
  keyless?: boolean;
  keyHint?: string;
  docs: string;
};

/**
 * Provider catalog. Model names were verified against the vendors' public
 * changelogs at build time (see README). The input field accepts anything,
 * so if a newer model ships you can paste it directly.
 *
 * Order matters: the first entry is the default tab when the dialog opens.
 * We lead with `qwen` because it runs locally through Ollama — no key, no
 * network egress, and nothing to pay for. The cloud providers follow.
 */
const PROVIDERS: ProviderDef[] = [
  {
    id: "qwen",
    label: "Qwen · local",
    // The coding-mxfp8 variant is Qwen 3.6 fine-tuned for coding and
    // quantised at mxfp8 (near-lossless, 38 GB). That's the best-quality
    // local option for .csd editing — beats the default Q4 tag and
    // matches cloud models more closely. Fallbacks below are listed for
    // users with less RAM. Static hints; the real dropdown is populated
    // from /api/qwen/status when Ollama is reachable.
    models: [
      "qwen3.6:35b-a3b-coding-mxfp8",
      "qwen3.6",
      "qwen3:14b",
      "qwen3:8b",
      "qwen2.5-coder:7b",
    ],
    local: true,
    docs: "https://ollama.com/library/qwen3.6",
  },
  {
    // Pollinations' free public endpoint (text.pollinations.ai). No signup,
    // no key, no credit card — the anonymous tier serves OpenAI's open-
    // weight GPT-OSS-20B via OVH under the model alias `openai`. Rate
    // limited to one request every ~15 seconds per IP; anything more
    // returns HTTP 429. Verified keyless as of 2026-04.
    id: "pollinations",
    label: "Pollinations · free",
    // `openai` is the default alias that resolves to the anonymous tier's
    // GPT-OSS-20B deployment. The dialog overwrites this list with the
    // live /api/pollinations/status response when the tab is opened.
    models: ["openai"],
    local: false,
    keyless: true,
    docs: "https://github.com/pollinations/pollinations/blob/master/APIDOCS.md",
  },
  {
    id: "openai",
    label: "OpenAI",
    models: ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.3-instant"],
    local: false,
    keyHint: "sk-…",
    docs: "https://platform.openai.com/api-keys",
  },
  {
    id: "anthropic",
    label: "Claude",
    models: [
      "claude-opus-4-7",
      "claude-sonnet-4-6",
      "claude-haiku-4-5-20251001",
    ],
    local: false,
    keyHint: "sk-ant-…",
    docs: "https://console.anthropic.com/settings/keys",
  },
  {
    id: "gemini",
    label: "Gemini",
    models: ["gemini-3.1-pro-preview", "gemini-3-flash-preview"],
    local: false,
    keyHint: "AIza…",
    docs: "https://aistudio.google.com/app/apikey",
  },
  {
    id: "openrouter",
    label: "OpenRouter · free",
    // `openrouter/free` randomly routes across free models. Users can also
    // select a pinned free variant by setting a specific `:free` model id.
    models: ["openrouter/free", "qwen/qwen3-coder:free", "openai/gpt-oss-20b:free"],
    local: false,
    keyHint: "sk-or-…",
    docs: "https://openrouter.ai/settings/keys",
  },
];

// Default selection when the dialog first opens. Kept as a module const so
// the initial useState seed and the provider-switch effect agree.
const DEFAULT_PROVIDER: LlmProvider = PROVIDERS[0].id;
const DEFAULT_MODEL: string = PROVIDERS[0].models[0];

export default function EditDialog({ open, runId, onClose, onEdited }: Props) {
  const [provider, setProvider] = useState<LlmProvider>(DEFAULT_PROVIDER);
  const [keys, setKeys] = useState<KeyStatusMap | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [model, setModel] = useState<string>(DEFAULT_MODEL);
  const [instruction, setInstruction] = useState("");
  const [busy, setBusy] = useState<null | "key" | "edit">(null);
  const [error, setError] = useState<string | null>(null);
  const [qwenStatus, setQwenStatus] = useState<QwenStatus | null>(null);
  const [pollinationsStatus, setPollinationsStatus] =
    useState<PollinationsStatus | null>(null);
  // Qwen-only: toggle Ollama's `think` flag (chain-of-thought). Default on
  // because the backend has always run it on, and the quality gap on Csound
  // edits is large. The checkbox is only rendered on the Qwen tab.
  const [deepThink, setDeepThink] = useState(true);
  const firstFieldRef = useRef<HTMLButtonElement | null>(null);

  const providerDef = PROVIDERS.find((p) => p.id === provider)!;
  const keyStatus =
    providerDef.local || providerDef.keyless
      ? undefined
      : keys?.[provider as KeyProvider];

  // Qwen is ready when the local Ollama server responded on the probe.
  const qwenReady = Boolean(qwenStatus?.available);
  // Pollinations is ready when text.pollinations.ai responded on the probe.
  const pollinationsReady = Boolean(pollinationsStatus?.available);
  const providerReady = providerDef.local
    ? qwenReady
    : providerDef.keyless
    ? pollinationsReady
    : Boolean(keyStatus?.present);

  // Reload the keys state whenever the dialog opens, and probe Ollama so
  // the Qwen tab can show an honest "ready / not reachable" badge.
  useEffect(() => {
    if (!open) return;
    setError(null);
    setKeyInput("");
    setShowKey(false);
    listKeys()
      .then(setKeys)
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Failed to load keys")
      );
    fetchQwenStatus()
      .then(setQwenStatus)
      .catch(() => setQwenStatus({ available: false, base_url: "", models: [] }));
    fetchPollinationsStatus()
      .then(setPollinationsStatus)
      .catch(() =>
        setPollinationsStatus({ available: false, base_url: "", models: [] })
      );
  }, [open]);

  // Re-probe when the user flips to the Qwen or Pollinations tab (cheap;
  // the endpoints themselves have a 2–3 s timeout). Gives instant feedback
  // if they just started Ollama or came back online.
  useEffect(() => {
    if (!open) return;
    if (provider === "qwen") {
      fetchQwenStatus()
        .then(setQwenStatus)
        .catch(() =>
          setQwenStatus({ available: false, base_url: "", models: [] })
        );
    } else if (provider === "pollinations") {
      fetchPollinationsStatus()
        .then(setPollinationsStatus)
        .catch(() =>
          setPollinationsStatus({ available: false, base_url: "", models: [] })
        );
    }
  }, [open, provider]);

  // When switching provider, default the model to that provider's top suggestion
  // unless the current value is already one of its listed models. For Qwen
  // we prefer whatever Ollama reports as actually installed, otherwise the
  // default wouldn't match a user's local library.
  useEffect(() => {
    const def = PROVIDERS.find((p) => p.id === provider)!;
    const suggestions = modelSuggestions(def, qwenStatus, pollinationsStatus);
    if (!suggestions.includes(model)) {
      setModel(suggestions[0] ?? def.models[0]);
    }
    setKeyInput("");
    setShowKey(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, qwenStatus, pollinationsStatus]);

  // Refs track the latest values so the keyboard-shortcut effect below can
  // read them without re-subscribing on every keystroke.
  const instructionRef = useRef(instruction);
  const runIdRef = useRef(runId);
  const submitRef = useRef<() => void>(() => {});
  instructionRef.current = instruction;
  runIdRef.current = runId;

  // Auto-focus the first provider tab — ONLY when the dialog opens, never on
  // re-render. Putting `instruction` in a combined effect's dep array caused
  // focus to jump back to this button on every keystroke.
  useEffect(() => {
    if (open) firstFieldRef.current?.focus();
  }, [open]);

  // Esc closes; ⌘/Ctrl-Enter submits. Bound once per open/close cycle; the
  // handler reads the latest instruction / runId / submit via refs.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (
        (e.metaKey || e.ctrlKey) &&
        e.key === "Enter" &&
        instructionRef.current.trim() &&
        runIdRef.current
      ) {
        e.preventDefault();
        submitRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const handleSaveKey = useCallback(async () => {
    if (!keyInput.trim()) return;
    // Save/delete only apply to key-backed providers; the UI that calls
    // these is gated behind `!providerDef.local`, so a narrowing cast is
    // safe and keeps the api helper signatures honest.
    if (providerDef.local) return;
    setBusy("key");
    setError(null);
    try {
      await saveKey(provider as KeyProvider, keyInput.trim());
      const next = await listKeys();
      setKeys(next);
      setKeyInput("");
      setShowKey(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setBusy(null);
    }
  }, [keyInput, provider, providerDef.local]);

  const handleDeleteKey = useCallback(async () => {
    if (providerDef.local) return;
    setBusy("key");
    setError(null);
    try {
      await deleteKey(provider as KeyProvider);
      const next = await listKeys();
      setKeys(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setBusy(null);
    }
  }, [provider, providerDef.local]);

  // Keep submitRef pointed at the latest submit closure so the keyboard
  // shortcut ⌘↵ always uses current form values without re-registering the
  // keydown listener on every keystroke.
  submitRef.current = () => {
    void submit();
  };

  async function submit() {
    if (!runId) return;
    if (providerDef.local) {
      if (!qwenReady) {
        setError(
          `Ollama is not reachable at ${qwenStatus?.base_url || "http://127.0.0.1:11434"}. ` +
            "Install it from ollama.com and run `ollama serve`."
        );
        return;
      }
    } else if (providerDef.keyless) {
      if (!pollinationsReady) {
        setError(
          `Pollinations is not reachable at ${
            pollinationsStatus?.base_url || "text.pollinations.ai"
          }. Check your internet connection.`
        );
        return;
      }
    } else if (!keyStatus?.present) {
      setError("Save an API key for this provider first.");
      return;
    }
    if (!instruction.trim()) {
      setError("Write an instruction first.");
      return;
    }
    setBusy("edit");
    setError(null);
    try {
      const result = await editRun({
        run_id: runId,
        provider,
        model: model.trim(),
        instruction: instruction.trim(),
        // Forward the reasoning toggle only for the local qwen provider;
        // the backend ignores it for anyone else anyway.
        ...(provider === "qwen" ? { think: deepThink } : {}),
      });
      onEdited(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Edit failed");
    } finally {
      setBusy(null);
    }
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/55 px-4 py-10 backdrop-blur-[1px]"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="edit-dialog-title"
    >
      <div className="relative flex max-h-[calc(100vh-3rem)] w-full max-w-[640px] flex-col overflow-hidden border border-ink bg-paper shadow-[8px_8px_0_rgba(24,22,26,0.12)]">
        {/* ——— Header ——— */}
        <div className="flex items-end justify-between border-b border-ink px-5 pb-3 pt-4">
          <div>
            <div className="eyebrow">Edit with an external model</div>
            <h2
              id="edit-dialog-title"
              className="display mt-1 text-[34px] leading-none"
            >
              Revise<span className="text-ink-red">.</span>
            </h2>
            <p className="mono mt-1 text-[11px] text-ink-muted">
              source · {runId ? `run #${runId.slice(-8)}` : "no run selected"}
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="mono -mr-1 px-2 py-1 text-[14px] text-ink-muted hover:text-ink-red"
          >
            ×
          </button>
        </div>

        {/* ——— Provider tabs ——— */}
        <div className="flex border-b border-rule-2">
          {PROVIDERS.map((p, i) => (
            <button
              key={p.id}
              ref={i === 0 ? firstFieldRef : undefined}
              onClick={() => setProvider(p.id)}
              className={
                "mono flex-1 truncate border-r border-rule-2 px-3 py-2 text-[11px] uppercase tracking-widest last:border-r-0 md:px-4 md:text-[12px] " +
                (provider === p.id
                  ? "bg-paper-3 text-ink"
                  : "bg-paper-2 text-ink-muted hover:text-ink")
              }
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* ——— Body ——— */}
        <div className="min-h-0 space-y-4 overflow-y-auto px-5 py-4">
          {providerDef.keyless ? (
            /* Pollinations · free public endpoint — no key, no local install.
               Show reachability + rate-limit reminder so the user isn't
               surprised by a 429 on the second click. */
            <section className="border border-rule-2 bg-paper-2">
              <div className="flex items-center justify-between border-b border-rule-2 px-3 py-2">
                <span className="eyebrow-ink">Public endpoint · Pollinations</span>
                <a
                  href={providerDef.docs}
                  target="_blank"
                  rel="noreferrer"
                  className="mono text-[10px] uppercase tracking-widest text-ink-red underline-offset-4 hover:underline"
                >
                  api docs →
                </a>
              </div>
              {pollinationsStatus == null ? (
                <div className="mono px-3 py-3 text-[12px] text-ink-muted">
                  probing pollinations…
                </div>
              ) : pollinationsReady ? (
                <div className="flex flex-wrap items-center gap-3 px-3 py-3">
                  <span
                    className="inline-block h-2 w-2 bg-signal"
                    aria-hidden
                  />
                  <span className="mono text-[12px]">
                    Pollinations ready ·{" "}
                    <span className="tabular">
                      {pollinationsStatus.base_url}
                    </span>
                  </span>
                  <span className="mono text-[10px] text-ink-muted">
                    anonymous tier · no signup · ~1 req /{" "}
                    {pollinationsStatus.rate_limit_seconds ?? 15}s per IP
                  </span>
                  <span className="mono ml-auto text-[11px] text-ink-muted">
                    {pollinationsStatus.models.length} model
                    {pollinationsStatus.models.length === 1 ? "" : "s"}{" "}
                    available
                  </span>
                </div>
              ) : (
                <div className="space-y-2 px-3 py-3">
                  <div className="flex items-center gap-3">
                    <span
                      className="inline-block h-2 w-2 bg-ink-red"
                      aria-hidden
                    />
                    <span className="mono text-[12px]">
                      Pollinations not reachable at{" "}
                      <span className="tabular">
                        {pollinationsStatus.base_url ||
                          "https://text.pollinations.ai"}
                      </span>
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        fetchPollinationsStatus()
                          .then(setPollinationsStatus)
                          .catch(() => {})
                      }
                      className="mono ml-auto border border-ink bg-paper px-2 py-1 text-[11px] uppercase tracking-widest hover:text-ink-red"
                    >
                      retry
                    </button>
                  </div>
                  {pollinationsStatus.error && (
                    <p className="mono text-[10px] text-ink-muted">
                      probe error: {pollinationsStatus.error}
                    </p>
                  )}
                </div>
              )}
            </section>
          ) : providerDef.local ? (
            /* Qwen · local Ollama — no API key needed. Show reachability. */
            <section className="border border-rule-2 bg-paper-2">
              <div className="flex items-center justify-between border-b border-rule-2 px-3 py-2">
                <span className="eyebrow-ink">Local runtime · Ollama</span>
                <a
                  href={providerDef.docs}
                  target="_blank"
                  rel="noreferrer"
                  className="mono text-[10px] uppercase tracking-widest text-ink-red underline-offset-4 hover:underline"
                >
                  qwen models →
                </a>
              </div>
              {qwenStatus == null ? (
                <div className="mono px-3 py-3 text-[12px] text-ink-muted">
                  probing ollama…
                </div>
              ) : qwenReady ? (
                <>
                  <div className="flex flex-wrap items-center gap-3 px-3 py-3">
                    <span
                      className="inline-block h-2 w-2 bg-signal"
                      aria-hidden
                    />
                    <span className="mono text-[12px]">
                      Ollama ready ·{" "}
                      <span className="tabular">{qwenStatus.base_url}</span>
                    </span>
                    <span className="mono text-[10px] text-ink-muted">
                      no key required · request never leaves your machine
                    </span>
                    <span className="mono ml-auto text-[11px] text-ink-muted">
                      {(qwenStatus.qwen_models?.length ?? 0)} qwen ·{" "}
                      {qwenStatus.models.length} total installed
                    </span>
                  </div>
                  {/* Deep-reasoning toggle. The backend forwards `think` to
                      Ollama's /api/chat; Qwen 3/3.6 models then run their
                      chain-of-thought before emitting the .csd. Non-reasoning
                      Qwen tags (qwen2.5-coder, qwen2.5) silently ignore the
                      flag, so leaving this on for them costs nothing. */}
                  <label className="flex cursor-pointer items-start gap-3 border-t border-rule-2 px-3 py-3">
                    <input
                      type="checkbox"
                      checked={deepThink}
                      onChange={(e) => setDeepThink(e.target.checked)}
                      className="mt-[3px] h-3 w-3 accent-ink"
                    />
                    <span className="flex-1">
                      <span className="mono text-[12px]">
                        Deep reasoning{" "}
                        <span className="text-ink-muted">
                          ({deepThink ? "on" : "off"})
                        </span>
                      </span>
                      <span className="mono mt-0.5 block text-[10px] text-ink-muted">
                        Qwen 3/3.6 chain-of-thought. Much better on non-trivial
                        edits, but each run takes ~1–3 min instead of ~15–25 s.
                        Ignored by non-reasoning tags (qwen2.5, qwen2.5-coder).
                      </span>
                    </span>
                  </label>
                </>
              ) : (
                <div className="space-y-2 px-3 py-3">
                  <div className="flex items-center gap-3">
                    <span
                      className="inline-block h-2 w-2 bg-ink-red"
                      aria-hidden
                    />
                    <span className="mono text-[12px]">
                      Ollama not reachable at{" "}
                      <span className="tabular">
                        {qwenStatus.base_url || "http://127.0.0.1:11434"}
                      </span>
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        fetchQwenStatus().then(setQwenStatus).catch(() => {})
                      }
                      className="mono ml-auto border border-ink bg-paper px-2 py-1 text-[11px] uppercase tracking-widest hover:text-ink-red"
                    >
                      retry
                    </button>
                  </div>
                  <pre className="mono bg-paper-3/40 overflow-x-auto px-3 py-2 text-[11px] leading-relaxed text-ink">
                    {`# install & run Ollama (one-time setup)
brew install ollama           # or see ollama.com for Linux / Windows
ollama serve                  # keep running in a terminal
ollama pull qwen2.5-coder:7b  # ~4 GB`}
                  </pre>
                  {qwenStatus.error && (
                    <p className="mono text-[10px] text-ink-muted">
                      probe error: {qwenStatus.error}
                    </p>
                  )}
                </div>
              )}
            </section>
          ) : (
            /* Cloud providers — require an API key. */
            <section className="border border-rule-2 bg-paper-2">
              <div className="flex items-center justify-between border-b border-rule-2 px-3 py-2">
                <span className="eyebrow-ink">
                  API key · {providerDef.label}
                </span>
                <a
                  href={providerDef.docs}
                  target="_blank"
                  rel="noreferrer"
                  className="mono text-[10px] uppercase tracking-widest text-ink-red underline-offset-4 hover:underline"
                >
                  get a key →
                </a>
              </div>
              <p className="mono border-b border-rule-2 px-3 py-2 text-[10px] text-ink-muted">
                This provider requires your own paid key. The app stores only a
                masked key status in UI and logs per-run token/cost metadata
                when available.
              </p>

              {keyStatus?.present ? (
                <div className="flex items-center gap-3 px-3 py-3">
                  <span
                    className="inline-block h-2 w-2 bg-signal"
                    aria-hidden
                  />
                  <span className="mono text-[12px]">
                    Stored ·{" "}
                    <span className="tabular">{keyStatus.masked}</span>
                  </span>
                  <span className="mono text-[10px] text-ink-muted">
                    kept in ~/.cstore/keys.json · never sent to the browser
                  </span>
                  <button
                    onClick={handleDeleteKey}
                    disabled={busy !== null}
                    className="mono ml-auto border border-ink bg-paper px-2 py-1 text-[11px] uppercase tracking-widest hover:text-ink-red disabled:opacity-40"
                  >
                    {busy === "key" ? "…" : "remove"}
                  </button>
                </div>
              ) : (
                <div className="flex items-stretch gap-2 px-3 py-3">
                  <input
                    type={showKey ? "text" : "password"}
                    value={keyInput}
                    onChange={(e) => setKeyInput(e.target.value)}
                    placeholder={providerDef.keyHint}
                    autoComplete="off"
                    spellCheck={false}
                    className="mono flex-1 border border-rule bg-paper px-2 py-1 text-[12px] outline-none focus:border-ink-red"
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && keyInput.trim()) {
                        e.preventDefault();
                        void handleSaveKey();
                      }
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey((v) => !v)}
                    aria-label={showKey ? "Hide key" : "Show key"}
                    className="mono border border-rule bg-paper px-2 py-1 text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink"
                  >
                    {showKey ? "hide" : "show"}
                  </button>
                  <button
                    type="button"
                    onClick={handleSaveKey}
                    disabled={!keyInput.trim() || busy !== null}
                    className="mono border border-ink bg-ink px-3 py-1 text-[11px] uppercase tracking-widest text-paper disabled:opacity-40"
                  >
                    {busy === "key" ? "saving" : "save"}
                  </button>
                </div>
              )}
            </section>
          )}

          {/* Model input — dropdown + free text so new model IDs work immediately. */}
          <section>
            <div className="mb-1 eyebrow-ink">Model</div>
            <div className="flex items-stretch gap-2">
              <input
                type="text"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                list={`models-${provider}`}
                className="mono flex-1 border border-rule bg-paper px-2 py-1 text-[12px] outline-none focus:border-ink-red"
                placeholder={providerDef.models[0]}
              />
              <datalist id={`models-${provider}`}>
                {modelSuggestions(providerDef, qwenStatus, pollinationsStatus).map(
                  (m) => (
                    <option key={m} value={m} />
                  )
                )}
              </datalist>
            </div>
            <p className="mono mt-1 text-[10px] text-ink-muted">
              {provider === "qwen"
                ? qwenReady && (qwenStatus?.qwen_models?.length ?? 0) > 0
                  ? `Installed qwen models: ${qwenStatus!
                      .qwen_models!.map((m) => m.name)
                      .join(" · ")}`
                  : "Type any Ollama model tag — e.g. qwen2.5-coder:7b. Pull it first with `ollama pull <name>`."
                : provider === "pollinations"
                ? pollinationsReady && pollinationsStatus!.models.length > 0
                  ? `Available (anonymous tier): ${pollinationsStatus!.models
                      .map((m) => m.name)
                      .join(" · ")} · currently GPT-OSS-20B`
                  : "Type any model ID from text.pollinations.ai/models — e.g. openai (GPT-OSS-20B)."
                : `Type any model ID. Suggestions: ${providerDef.models.join(" · ")}`}
            </p>
          </section>

          {/* Instruction */}
          <section>
            <div className="mb-1 flex items-baseline justify-between">
              <span className="eyebrow-ink">Instruction</span>
              <span className="mono text-[10px] text-ink-muted">
                what should the model change?
              </span>
            </div>
            <div className="relative">
              <textarea
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                rows={4}
                placeholder={`e.g. "Add a long plate reverb to all voices."\n     "Transpose down one octave, keep the rhythm."\n     "Replace the FM voice with a plucked string."`}
                className="mono scroll-ink block max-h-[40vh] min-h-[7rem] w-full resize-y border border-rule bg-paper px-3 py-2 text-[13px] leading-relaxed outline-none focus:border-ink-red"
              />
              {busy === "edit" && (
                <div className="hatch pointer-events-none absolute inset-0 border border-rule" />
              )}
            </div>
            <p className="mono mt-1 text-[10px] text-ink-muted">
              The current .csd is attached automatically. The model returns a
              full replacement .csd, which we render with Csound and save as a
              new run.
            </p>
          </section>

          {/* Error strip */}
          {error && (
            <div className="mono border border-ink-red bg-paper-2 px-3 py-2 text-[12px] text-ink-red-2">
              {error}
            </div>
          )}
        </div>

        {/* ——— Footer ——— */}
        <div className="flex items-center justify-between gap-3 border-t border-ink bg-paper-2 px-5 py-3">
          <span className="mono text-[10px] text-ink-muted">
            ⌘↵ to send · Esc to close
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="mono border border-rule bg-paper px-3 py-1 text-[11px] uppercase tracking-widest hover:border-ink"
            >
              cancel
            </button>
            <button
              onClick={submit}
              disabled={
                !runId ||
                !instruction.trim() ||
                !providerReady ||
                busy !== null
              }
              className="mono flex items-center gap-2 border border-ink bg-ink px-3 py-1 text-[11px] uppercase tracking-widest text-paper disabled:opacity-40"
            >
              {busy === "edit"
                ? "sending…"
                : providerDef.local
                ? "run · ollama"
                : providerDef.keyless
                ? "send · pollinations"
                : `send · ${providerDef.label.toLowerCase()}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Merge a provider's static model suggestions with any live data we have:
 *   - Qwen: what Ollama reports as actually installed locally
 *   - Pollinations: which models the anonymous tier can actually call
 * Keeps the live list first so the user sees a working list immediately.
 */
function modelSuggestions(
  def: ProviderDef,
  qwenStatus: QwenStatus | null,
  pollinationsStatus: PollinationsStatus | null
): string[] {
  const merge = (live: string[]) => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const name of [...live, ...def.models]) {
      if (!seen.has(name)) {
        seen.add(name);
        out.push(name);
      }
    }
    return out;
  };
  if (def.id === "qwen") {
    return merge((qwenStatus?.qwen_models ?? []).map((m) => m.name));
  }
  if (def.id === "pollinations") {
    // Include both canonical names and their aliases so users can type
    // either "openai" (the default anonymous alias) or the concrete
    // "openai-fast" / "gpt-oss-20b" deployment name.
    const live = (pollinationsStatus?.models ?? []).flatMap((m) => [
      m.name,
      ...(m.aliases || []),
    ]);
    return merge(live);
  }
  return def.models;
}
