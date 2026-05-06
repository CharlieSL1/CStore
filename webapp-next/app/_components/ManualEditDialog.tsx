"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { renderCsd, type RenderResponse } from "../_lib/api";

type Props = {
  open: boolean;
  /** Source run_id to stamp into the new run's `derived_from` meta. */
  runId: string | null;
  /** Current .csd text from the console — seeded into the editor on open. */
  initialCsd: string;
  onClose: () => void;
  /** Fired with the rendered run so the console can load audio + source. */
  onRendered: (result: RenderResponse) => void;
};

/**
 * Manual-edit window. Lets the user hand-tweak the .csd that came out of the
 * generator (or a previous LLM edit) and re-run Csound without touching any
 * language model. The /api/render endpoint does the actual work: it saves
 * the text as a new run, renders it with the csound binary, and the result
 * flows back through `onRendered`.
 *
 * Keeps scope small on purpose: no syntax highlighting, no autocomplete.
 * A monospace textarea with line numbers is enough for in-session tweaks;
 * serious editing still happens in an external editor or the Csound IDE.
 */
export default function ManualEditDialog({
  open,
  runId,
  initialCsd,
  onClose,
  onRendered,
}: Props) {
  const [text, setText] = useState<string>(initialCsd);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const gutterRef = useRef<HTMLPreElement | null>(null);

  // Re-seed the editor each time the dialog opens so the user sees the
  // current .csd, not a stale copy from a previous session.
  useEffect(() => {
    if (!open) return;
    setText(initialCsd);
    setError(null);
    // Defer focus so the textarea is actually in the DOM.
    const t = window.setTimeout(() => textareaRef.current?.focus(), 0);
    return () => window.clearTimeout(t);
  }, [open, initialCsd]);

  // Keep the line-number gutter in vertical sync with the textarea's scroll.
  const handleScroll = useCallback(() => {
    if (gutterRef.current && textareaRef.current) {
      gutterRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  }, []);

  const lineNumbers = useMemo(() => {
    const count = Math.max(1, text.split("\n").length);
    return Array.from({ length: count }, (_, i) => String(i + 1).padStart(3, " "))
      .join("\n");
  }, [text]);

  const dirty = text !== initialCsd;
  const canRender =
    !busy &&
    text.trim().length > 0 &&
    text.includes("<CsoundSynthesizer>") &&
    text.includes("</CsoundSynthesizer>");

  const submit = useCallback(async () => {
    if (!canRender) return;
    setBusy(true);
    setError(null);
    try {
      const result = await renderCsd({
        csd: text,
        derived_from: runId,
      });
      onRendered(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Render failed");
    } finally {
      setBusy(false);
    }
  }, [canRender, text, runId, onRendered]);

  // Esc closes; ⌘/Ctrl-Enter renders. Bound once per open/close cycle.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const onTextareaKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        void submit();
        return;
      }
      // Tab inserts two spaces instead of moving focus — otherwise it's
      // nearly impossible to edit a .csd where indentation matters.
      if (e.key === "Tab") {
        e.preventDefault();
        const el = e.currentTarget;
        const { selectionStart, selectionEnd, value } = el;
        const insert = "  ";
        const next =
          value.slice(0, selectionStart) + insert + value.slice(selectionEnd);
        setText(next);
        // Restore cursor after React re-renders the controlled value.
        window.requestAnimationFrame(() => {
          el.selectionStart = el.selectionEnd = selectionStart + insert.length;
        });
      }
    },
    [submit]
  );

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/55 px-4 py-6 backdrop-blur-[1px]"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="manual-edit-title"
    >
      <div className="relative flex max-h-[calc(100vh-3rem)] w-full max-w-[960px] flex-col overflow-hidden border border-ink bg-paper shadow-[8px_8px_0_rgba(24,22,26,0.12)]">
        {/* ——— Header ——— */}
        <div className="flex items-end justify-between border-b border-ink px-5 pb-3 pt-4">
          <div>
            <div className="eyebrow">Edit the source by hand</div>
            <h2
              id="manual-edit-title"
              className="display mt-1 text-[34px] leading-none"
            >
              Rewrite<span className="text-ink-red">.</span>
            </h2>
            <p className="mono mt-1 text-[11px] text-ink-muted">
              source · {runId ? `run #${runId.slice(-8)}` : "no run selected"}
              {dirty && (
                <span className="ml-2 text-ink-red">· modified</span>
              )}
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

        {/* ——— Editor ——— */}
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-5 py-4">
          <div className="flex items-center justify-between">
            <span className="eyebrow-ink">output.csd · editable</span>
            <div className="mono tabular flex items-center gap-3 text-[11px] text-ink-muted">
              <span>{text.split("\n").length} lines</span>
              <span>·</span>
              <span>{text.length} chars</span>
              <button
                type="button"
                onClick={() => setText(initialCsd)}
                disabled={!dirty || busy}
                className="mono border border-rule bg-paper px-2 py-1 text-[10px] uppercase tracking-widest hover:border-ink disabled:pointer-events-none disabled:opacity-40"
              >
                revert
              </button>
            </div>
          </div>

          {/* Gutter + textarea share a row so the line numbers align with
              the text. The gutter scrolls programmatically in sync with the
              textarea (see handleScroll above). */}
          <div className="scroll-ink grid max-h-[60vh] min-h-[320px] grid-cols-[3rem_1fr] overflow-hidden border border-ink bg-paper-2">
            <pre
              ref={gutterRef}
              aria-hidden
              className="mono tabular select-none overflow-hidden whitespace-pre border-r border-rule-2 bg-paper-3/40 py-3 text-right text-[11px] leading-[1.55] text-ink-muted"
            >
              {lineNumbers}
            </pre>
            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onScroll={handleScroll}
              onKeyDown={onTextareaKeyDown}
              spellCheck={false}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              className="mono scroll-ink block w-full resize-none whitespace-pre overflow-auto overscroll-contain bg-paper-2 px-3 py-3 text-[12px] leading-[1.55] text-ink outline-none focus:bg-paper"
              placeholder="<CsoundSynthesizer>…</CsoundSynthesizer>"
            />
          </div>

          <p className="mono text-[10px] text-ink-muted">
            Edits are rendered locally by the Csound binary — no model is
            invoked. A successful render is saved as a new library entry
            tagged <span className="text-ink">manual_edit</span>.
          </p>

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
            ⌘↵ to render · Esc to close · Tab inserts two spaces
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="mono border border-rule bg-paper px-3 py-1 text-[11px] uppercase tracking-widest hover:border-ink"
            >
              cancel
            </button>
            <button
              onClick={() => void submit()}
              disabled={!canRender}
              className="mono flex items-center gap-2 border border-ink bg-ink px-3 py-1 text-[11px] uppercase tracking-widest text-paper disabled:opacity-40"
            >
              {busy ? "rendering…" : "render with csound"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
