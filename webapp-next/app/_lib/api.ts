/**
 * Thin wrappers around the Flask backend in ../webapp/app.py.
 * Paths are relative so Next.js's rewrites (see next.config.ts) can proxy them.
 */

export type GenerateResponse = {
  success: true;
  csd: string;
  run_id: string;
  csd_url: string;
  wav_url: string;
  checkpoint?: string;
};

export type ModelEntry = {
  path: string; // relative label, e.g. "Cstore_V1.0.1/best"
  absolute: string;
  family: string;
  variant: string;
  label: string;
  is_default: boolean;
};

export type ModelsResponse = {
  root: string;
  active: string;
  models: ModelEntry[];
};

export type GenerateError = {
  success?: false;
  error: string;
};

export type RunEntry = {
  run_id: string;
  has_csd: boolean;
  has_wav: boolean;
};

export type RunMeta = {
  kind?: "edit" | "manual_edit";
  derived_from?: string;
  provider?: LlmProvider;
  model?: string;
  instruction?: string;
  render_ok?: boolean;
  /** Set by the user via the star button in the library. Persisted in
   *  meta.json alongside any existing edit/manual_edit provenance. */
  favorite?: boolean;
};

export type RunEntryWithMeta = RunEntry & { meta?: RunMeta };

export type ListResponse = {
  runs: RunEntryWithMeta[];
};

export type ConsoleLevel = "info" | "csound" | "warn" | "err" | "ok" | "sys";

export type ConsoleEntry = {
  seq: number;
  t: number; // unix seconds, fractional
  level: ConsoleLevel;
  run_id: string | null;
  text: string;
};

export type ConsoleResponse = {
  seq: number;
  entries: ConsoleEntry[];
  started_at: number;
};

/** Providers that require a user-supplied API key. */
export type KeyProvider = "openai" | "anthropic" | "gemini";
/** All providers the backend's /api/edit endpoint accepts. `qwen` runs on a
 *  local Ollama instance and is keyless. `pollinations` calls a free public
 *  anonymous-tier endpoint at text.pollinations.ai (GPT-OSS-20B, ~15s rate
 *  limit). Both are keyless; everything else stores a key locally. */
export type LlmProvider = KeyProvider | "qwen" | "pollinations";

export type KeyStatus = { present: boolean; masked: string };
/** Only the key-requiring providers appear here. */
export type KeyStatusMap = Record<KeyProvider, KeyStatus>;

export type OllamaModel = { name: string; size: number };
export type QwenStatus = {
  available: boolean;
  base_url: string;
  error?: string;
  models: OllamaModel[];
  qwen_models?: OllamaModel[];
};

export type PollinationsModel = {
  name: string;
  description: string;
  reasoning: boolean;
  aliases: string[];
};
export type PollinationsStatus = {
  available: boolean;
  base_url: string;
  error?: string;
  models: PollinationsModel[];
  rate_limit_seconds?: number;
};

export type EditResponse = {
  success: true;
  csd: string;
  run_id: string;
  csd_url: string;
  wav_url: string;
  derived_from: string;
  provider: LlmProvider;
  model: string;
  instruction: string;
};

export type RenderResponse = {
  success: true;
  csd: string;
  run_id: string;
  csd_url: string;
  wav_url: string;
  derived_from: string | null;
};

export async function generate(
  opts: { seed?: number; checkpoint?: string } = {}
): Promise<GenerateResponse> {
  const payload: Record<string, unknown> = {};
  if (opts.seed != null) payload.seed = opts.seed;
  if (opts.checkpoint) payload.checkpoint = opts.checkpoint;
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let data: GenerateResponse | GenerateError = { error: "" } as GenerateError;
  try {
    data = await res.json();
  } catch {
    // leave as default
  }

  if (!res.ok || !("success" in data) || !data.success) {
    const msg =
      "error" in data && data.error ? data.error : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

export async function listRuns(): Promise<RunEntryWithMeta[]> {
  const res = await fetch("/api/list", { cache: "no-store" });
  if (!res.ok) throw new Error(`List failed (${res.status})`);
  const data = (await res.json()) as ListResponse;
  return data.runs ?? [];
}

export async function fetchCsd(runId: string): Promise<string> {
  const res = await fetch(`/generated/${encodeURIComponent(runId)}/output.csd`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("File not found");
  return res.text();
}

export async function fetchConsole(
  since = 0,
  limit = 500
): Promise<ConsoleResponse> {
  const params = new URLSearchParams({
    since: String(since),
    limit: String(limit),
  });
  const res = await fetch(`/api/console?${params.toString()}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Console fetch failed (${res.status})`);
  return (await res.json()) as ConsoleResponse;
}

export async function listModels(): Promise<ModelsResponse> {
  const res = await fetch("/api/models", { cache: "no-store" });
  if (!res.ok) throw new Error(`Models list failed (${res.status})`);
  return (await res.json()) as ModelsResponse;
}

export async function setActiveModel(path: string): Promise<string> {
  const res = await fetch("/api/model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  let body: { active?: string; error?: string } = {};
  try {
    body = await res.json();
  } catch {
    // fall through
  }
  if (!res.ok) throw new Error(body.error || `Load failed (${res.status})`);
  return body.active ?? path;
}

// -------- LLM editing ------------------------------------------------------
//
// The API key NEVER travels through the browser: it's stored server-side at
// ~/.cstore/keys.json (chmod 0600) and only masked summaries come back here.
// All outbound calls to OpenAI / Anthropic / Gemini happen inside the Python
// sidecar using that stored key.

export async function listKeys(): Promise<KeyStatusMap> {
  const res = await fetch("/api/keys", { cache: "no-store" });
  if (!res.ok) throw new Error(`Keys list failed (${res.status})`);
  return (await res.json()) as KeyStatusMap;
}

export async function saveKey(
  provider: KeyProvider,
  key: string
): Promise<string> {
  const res = await fetch("/api/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, key }),
  });
  let body: { ok?: boolean; masked?: string; error?: string } = {};
  try {
    body = await res.json();
  } catch {
    /* keep defaults */
  }
  if (!res.ok || !body.ok) {
    throw new Error(body.error || `Save failed (${res.status})`);
  }
  return body.masked ?? "";
}

export async function deleteKey(provider: KeyProvider): Promise<void> {
  const res = await fetch("/api/keys", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!res.ok) {
    let msg = `Delete failed (${res.status})`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) msg = body.error;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
}

export async function fetchQwenStatus(): Promise<QwenStatus> {
  const res = await fetch("/api/qwen/status", { cache: "no-store" });
  if (!res.ok) throw new Error(`Qwen status failed (${res.status})`);
  return (await res.json()) as QwenStatus;
}

/**
 * Probe the free Pollinations endpoint (text.pollinations.ai).
 * Lists the subset of models that the `anonymous` tier can actually call
 * without an account so the dialog only ever offers reachable options.
 */
export async function fetchPollinationsStatus(): Promise<PollinationsStatus> {
  const res = await fetch("/api/pollinations/status", { cache: "no-store" });
  if (!res.ok) throw new Error(`Pollinations status failed (${res.status})`);
  return (await res.json()) as PollinationsStatus;
}

export async function editRun(opts: {
  run_id: string;
  provider: LlmProvider;
  model: string;
  instruction: string;
  /** Local-qwen only. When true (default on the backend) Ollama's `think`
   *  flag enables Qwen 3/3.6 chain-of-thought: slower, noticeably higher
   *  quality on non-trivial edits. Ignored by cloud providers. */
  think?: boolean;
}): Promise<EditResponse> {
  const res = await fetch("/api/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  let body:
    | EditResponse
    | { success?: false; error?: string; run_id?: string; csd_url?: string } = {
    error: "",
  };
  try {
    body = await res.json();
  } catch {
    /* defaults */
  }
  if (!res.ok || !("success" in body) || !body.success) {
    const msg =
      "error" in body && body.error
        ? body.error
        : `Edit failed (${res.status})`;
    throw new Error(msg);
  }
  return body;
}

/**
 * Toggle (or explicitly set) the favourite flag on a run. Favourites stay
 * pinned in the library drawer even after the 50-row recent cap has rolled
 * them out of the main scroll. The backend writes the flag into the run's
 * meta.json so it survives restarts.
 */
export async function setFavorite(
  runId: string,
  favorite?: boolean
): Promise<RunMeta> {
  const res = await fetch("/api/favorite", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_id: runId,
      ...(favorite != null ? { favorite } : {}),
    }),
  });
  let body: { ok?: boolean; meta?: RunMeta; error?: string } = {};
  try {
    body = await res.json();
  } catch {
    /* defaults */
  }
  if (!res.ok || !body.ok || !body.meta) {
    throw new Error(body.error || `Favorite toggle failed (${res.status})`);
  }
  return body.meta;
}

/**
 * Submit hand-edited .csd text and have the backend render it with Csound.
 * Creates a new run tagged `manual_edit` so it shows up in the library just
 * like a generation or LLM edit, but never touches any model.
 */
export async function renderCsd(opts: {
  csd: string;
  derived_from?: string | null;
}): Promise<RenderResponse> {
  const res = await fetch("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      csd: opts.csd,
      ...(opts.derived_from ? { derived_from: opts.derived_from } : {}),
    }),
  });
  let body:
    | RenderResponse
    | { success?: false; error?: string; run_id?: string; csd_url?: string } = {
    error: "",
  };
  try {
    body = await res.json();
  } catch {
    /* defaults */
  }
  if (!res.ok || !("success" in body) || !body.success) {
    const msg =
      "error" in body && body.error
        ? body.error
        : `Render failed (${res.status})`;
    throw new Error(msg);
  }
  return body;
}

/** Parse the run_id timestamp prefix (YYYYMMDD_HHMMSS_xxxxxxxx). */
export function formatRunId(runId: string): { date: string; time: string; hash: string } {
  const m = runId.match(
    /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_([0-9a-f]+)$/i
  );
  if (!m) return { date: "—", time: "—", hash: runId.slice(0, 8) };
  return {
    date: `${m[1]}-${m[2]}-${m[3]}`,
    time: `${m[4]}:${m[5]}:${m[6]}`,
    hash: m[7].slice(0, 8),
  };
}
