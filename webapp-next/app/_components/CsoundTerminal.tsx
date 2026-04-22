"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  fetchConsole,
  type ConsoleEntry,
  type ConsoleLevel,
} from "../_lib/api";

/**
 * Csound terminal — a live tail of everything the Python sidecar and the
 * csound binary are saying: sampling, attempts, the actual csound CLI
 * invocation, csound's per-instrument messages, the RMS watchdog check,
 * LLM edits, errors. Clients poll /api/console with a monotonic `since`
 * cursor so each request only carries new lines.
 *
 * Polling (instead of SSE) keeps the Flask sidecar trivially simple and
 * plays nicely with Next.js's rewrite proxy.
 */
const POLL_ACTIVE_MS = 500;  // while csound/generate is working
const POLL_IDLE_MS = 2000;   // nothing new for a while
const ACTIVE_WINDOW_MS = 5000; // recency window that counts as "active"
const MAX_RETAINED = 2000;   // client-side cap; matches server ring buffer

type FilterMode = "all" | "csound" | "events";

export default function CsoundTerminal() {
  const [entries, setEntries] = useState<ConsoleEntry[]>([]);
  const [connected, setConnected] = useState<boolean>(true);
  const [paused, setPaused] = useState<boolean>(false);
  const [follow, setFollow] = useState<boolean>(true);
  const [filter, setFilter] = useState<FilterMode>("all");

  const seqRef = useRef<number>(0);
  const lastActivityRef = useRef<number>(0);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const pausedRef = useRef<boolean>(paused);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  // ——— Poll loop ——————————————————————————————————————————————————
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const tick = async () => {
      if (cancelled) return;
      if (pausedRef.current) {
        timer = window.setTimeout(tick, POLL_IDLE_MS);
        return;
      }
      try {
        const data = await fetchConsole(seqRef.current, 500);
        if (cancelled) return;
        setConnected(true);

        // The server buffer is bounded, so if it wrapped past our cursor we
        // replay the whole snapshot (harmless at 2k entries).
        if (data.entries.length > 0) {
          lastActivityRef.current = Date.now();
          seqRef.current = data.seq;
          setEntries((prev) => {
            const merged = prev.concat(data.entries);
            return merged.length > MAX_RETAINED
              ? merged.slice(merged.length - MAX_RETAINED)
              : merged;
          });
        } else if (seqRef.current === 0) {
          // First poll on a quiet backend — at least sync the cursor.
          seqRef.current = data.seq;
        }
      } catch {
        if (!cancelled) setConnected(false);
      } finally {
        if (!cancelled) {
          const wait =
            Date.now() - lastActivityRef.current < ACTIVE_WINDOW_MS
              ? POLL_ACTIVE_MS
              : POLL_IDLE_MS;
          timer = window.setTimeout(tick, wait);
        }
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  // ——— Auto-scroll on new entries (only when follow is on) ————————
  useLayoutEffect(() => {
    if (!follow) return;
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [entries, follow]);

  // If the user scrolls up, pause follow. If they scroll back to the bottom,
  // resume. This matches what developers expect from a terminal.
  const onScroll = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const atBottom =
      el.scrollHeight - el.clientHeight - el.scrollTop < 4;
    setFollow(atBottom);
  }, []);

  const clear = useCallback(() => {
    setEntries([]);
  }, []);

  const jumpToEnd = useCallback(() => {
    setFollow(true);
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  const visible = useMemo(() => {
    if (filter === "all") return entries;
    if (filter === "csound")
      return entries.filter((e) => e.level === "csound");
    // "events" = everything except raw csound lines
    return entries.filter((e) => e.level !== "csound");
  }, [entries, filter]);

  const counts = useMemo(() => {
    let warn = 0;
    let err = 0;
    for (const e of entries) {
      if (e.level === "warn") warn += 1;
      else if (e.level === "err") err += 1;
    }
    return { warn, err, total: entries.length };
  }, [entries]);

  return (
    <div className="col-span-12 border border-ink bg-ink text-paper">
      {/* ——— Header strip ——— */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-paper/20 px-3 py-2">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className={
              "inline-block h-2 w-2 " +
              (!connected
                ? "bg-ink-red"
                : Date.now() - lastActivityRef.current < ACTIVE_WINDOW_MS
                ? "bg-signal"
                : "bg-paper/40")
            }
          />
          <span className="eyebrow !text-paper/70">Csound Terminal</span>
          <span className="mono tabular text-[11px] text-paper/50">
            {connected ? "connected · :5000" : "backend unreachable"}
          </span>
        </div>

        <div className="flex items-center gap-1">
          <FilterButton
            label="all"
            active={filter === "all"}
            onClick={() => setFilter("all")}
          />
          <FilterButton
            label="csound"
            active={filter === "csound"}
            onClick={() => setFilter("csound")}
          />
          <FilterButton
            label="events"
            active={filter === "events"}
            onClick={() => setFilter("events")}
          />

          <span className="mx-2 h-4 w-px bg-paper/20" aria-hidden />

          <button
            onClick={() => setPaused((p) => !p)}
            className="mono text-[10px] uppercase tracking-widest text-paper/70 hover:text-paper"
            title={paused ? "Resume polling" : "Pause polling"}
          >
            {paused ? "▶ resume" : "❚❚ pause"}
          </button>
          <button
            onClick={clear}
            className="mono ml-2 text-[10px] uppercase tracking-widest text-paper/70 hover:text-paper"
            title="Clear view (server buffer is untouched)"
          >
            clear
          </button>
          <button
            onClick={jumpToEnd}
            className={
              "mono ml-2 text-[10px] uppercase tracking-widest " +
              (follow ? "text-paper/30" : "text-paper hover:text-ink-red")
            }
            title="Jump to tail"
          >
            {follow ? "tail ✓" : "↓ jump"}
          </button>
        </div>
      </div>

      {/* ——— Log pane ——— */}
      <div
        ref={scrollerRef}
        onScroll={onScroll}
        className="scroll-ink mono max-h-[360px] min-h-[220px] overflow-auto whitespace-pre-wrap break-words p-3 text-[11px] leading-[1.55]"
      >
        {visible.length === 0 ? (
          <div className="text-paper/40">
            # awaiting backend output — press <span className="kbd">G</span>{" "}
            to draft a render and csound messages will stream here.
          </div>
        ) : (
          visible.map((e) => <Line key={e.seq} entry={e} />)
        )}
      </div>

      {/* ——— Footer: counts ——— */}
      <div className="flex items-center justify-between border-t border-paper/20 px-3 py-1.5">
        <span className="mono tabular text-[10px] text-paper/50">
          {counts.total} lines · {entries.length >= MAX_RETAINED ? "buffer full · " : ""}
          {paused ? "paused" : "live"}
        </span>
        <span className="mono tabular text-[10px] text-paper/50">
          {counts.warn > 0 && (
            <span className="mr-3 text-[#e0b341]">warn {counts.warn}</span>
          )}
          {counts.err > 0 && (
            <span className="text-ink-red">err {counts.err}</span>
          )}
        </span>
      </div>
    </div>
  );
}

function Line({ entry }: { entry: ConsoleEntry }) {
  const ts = formatTime(entry.t);
  const { color, tag } = levelStyle(entry.level);
  const runTag = entry.run_id ? entry.run_id.slice(-8) : "";
  return (
    <div className="flex gap-2">
      <span className="tabular w-[72px] shrink-0 text-paper/35">{ts}</span>
      <span className={"w-[54px] shrink-0 uppercase " + color}>{tag}</span>
      {runTag ? (
        <span className="w-[80px] shrink-0 text-paper/40">#{runTag}</span>
      ) : (
        <span className="w-[80px] shrink-0 text-paper/20">—</span>
      )}
      <span className={"flex-1 " + (entry.level === "err" ? "text-ink-red" : "text-paper/90")}>
        {entry.text || "\u00A0"}
      </span>
    </div>
  );
}

function FilterButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "mono px-2 py-0.5 text-[10px] uppercase tracking-widest " +
        (active
          ? "bg-paper text-ink"
          : "text-paper/60 hover:text-paper")
      }
    >
      {label}
    </button>
  );
}

function levelStyle(level: ConsoleLevel): { color: string; tag: string } {
  switch (level) {
    case "err":
      return { color: "text-ink-red", tag: "err" };
    case "warn":
      return { color: "text-[#e0b341]", tag: "warn" };
    case "ok":
      return { color: "text-signal", tag: "ok" };
    case "csound":
      return { color: "text-paper/60", tag: "csd" };
    case "sys":
      return { color: "text-paper", tag: "sys" };
    case "info":
    default:
      return { color: "text-paper/70", tag: "info" };
  }
}

function formatTime(t: number): string {
  const d = new Date(t * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss}.${ms.slice(0, 2)}`;
}
