"use client";

import { useEffect, useMemo, useState } from "react";
import {
  listModels,
  setActiveModel,
  type ModelEntry,
  type ModelsResponse,
} from "../_lib/api";

type Props = {
  /** Notify the parent when the active model changes (absolute path). */
  onActiveChange?: (active: string, selection: string | null) => void;
  /** Notify the parent so it can display transient errors in the status strip. */
  onError?: (msg: string) => void;
  /** Notify the parent that a model swap is in progress (to show a status). */
  onLoadingChange?: (busy: boolean, label: string | null) => void;
};

/**
 * Horizontal "patch bay" strip for choosing a model:
 *   [ MODEL ]  [ Cstore_V1.0.1 / best  ▾ ]  [ path ___________ ] [ load ]
 * Discovered checkpoints come from GET /api/models; the user can also paste
 * any absolute path on disk. Changing the dropdown sends the choice to the
 * backend immediately; the custom path is applied on "load".
 */
export default function ModelPicker({
  onActiveChange,
  onError,
  onLoadingChange,
}: Props) {
  const [state, setState] = useState<ModelsResponse | null>(null);
  const [selection, setSelection] = useState<string>("__default__");
  const [customPath, setCustomPath] = useState<string>("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listModels()
      .then((r) => {
        if (cancelled) return;
        setState(r);
        const match = r.models.find((m) => m.absolute === r.active);
        setSelection(match ? match.path : "__custom__");
        if (!match) setCustomPath(r.active);
        onActiveChange?.(r.active, match ? match.path : null);
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : "Failed to load model list";
        onError?.(msg);
      });
    return () => {
      cancelled = true;
    };
    // Run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeLabel = useMemo(() => {
    if (!state) return "—";
    const m = state.models.find((x) => x.absolute === state.active);
    return m ? m.label : shortenPath(state.active);
  }, [state]);

  async function applyPath(path: string, newSelection: string) {
    if (!path) return;
    setBusy(true);
    onLoadingChange?.(true, path);
    try {
      const active = await setActiveModel(path);
      setState((prev) => (prev ? { ...prev, active } : prev));
      setSelection(newSelection);
      const match = state?.models.find((m) => m.absolute === active);
      onActiveChange?.(active, match ? match.path : null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to switch model";
      onError?.(msg);
    } finally {
      setBusy(false);
      onLoadingChange?.(false, null);
    }
  }

  const groupedByFamily = useMemo(() => {
    if (!state) return new Map<string, ModelEntry[]>();
    const g = new Map<string, ModelEntry[]>();
    for (const m of state.models) {
      const list = g.get(m.family) ?? [];
      list.push(m);
      g.set(m.family, list);
    }
    return g;
  }, [state]);

  return (
    <div className="mt-4 grid grid-cols-12 gap-0 border border-ink bg-paper-2">
      <div className="col-span-12 flex items-center gap-3 border-b border-rule-2 px-3 py-2 md:col-span-auto md:border-b-0 md:border-r">
        <span className="eyebrow-ink">Model</span>
        <span className="mono text-[11px] text-ink-muted">
          active: <span className="text-ink">{activeLabel}</span>
        </span>
      </div>

      {/* Preset dropdown */}
      <div className="col-span-12 flex items-center gap-2 border-b border-rule-2 px-3 py-2 md:col-span-5 md:border-b-0 md:border-r">
        <span className="eyebrow">preset</span>
        <select
          value={selection}
          disabled={!state || busy}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "__custom__") {
              setSelection("__custom__");
              return;
            }
            const m = state?.models.find((x) => x.path === v);
            if (m) applyPath(m.path, v);
          }}
          className="mono tabular flex-1 border border-rule bg-paper px-2 py-1 text-[12px] outline-none focus:border-ink-red"
        >
          {state ? (
            <>
              {Array.from(groupedByFamily.entries()).map(([family, list]) => (
                <optgroup key={family} label={family}>
                  {list.map((m) => (
                    <option key={m.path} value={m.path}>
                      {m.variant}
                      {m.is_default ? "  ·  default" : ""}
                    </option>
                  ))}
                </optgroup>
              ))}
              <option value="__custom__">Custom path…</option>
            </>
          ) : (
            <option>loading…</option>
          )}
        </select>
      </div>

      {/* Custom path */}
      <div className="col-span-12 flex items-center gap-2 px-3 py-2 md:col-span-6">
        <span className="eyebrow">path</span>
        <input
          type="text"
          value={customPath}
          onChange={(e) => setCustomPath(e.target.value)}
          placeholder="/absolute/path/to/checkpoint  ·  or  Cstore_V1.0.2/best"
          className="mono flex-1 border border-rule bg-paper px-2 py-1 text-[12px] outline-none focus:border-ink-red"
          onKeyDown={(e) => {
            if (e.key === "Enter" && customPath.trim()) {
              applyPath(customPath.trim(), "__custom__");
            }
          }}
        />
        <button
          disabled={!customPath.trim() || busy}
          onClick={() => applyPath(customPath.trim(), "__custom__")}
          className="mono border border-ink bg-ink px-3 py-1 text-[11px] uppercase tracking-widest text-paper disabled:opacity-40"
        >
          {busy ? "loading" : "load"}
        </button>
      </div>
    </div>
  );
}

/** Keep long absolute paths readable in the "active: …" line. */
function shortenPath(p: string): string {
  const parts = p.split("/");
  if (parts.length <= 4) return p;
  return `…/${parts.slice(-3).join("/")}`;
}
