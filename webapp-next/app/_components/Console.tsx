"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchCsd,
  formatRunId,
  generate,
  listRuns,
  setFavorite,
  type EditResponse,
  type RenderResponse,
  type RunEntryWithMeta,
} from "../_lib/api";
import CsoundTerminal from "./CsoundTerminal";
import EditDialog from "./EditDialog";
import ManualEditDialog from "./ManualEditDialog";
import ModelPicker from "./ModelPicker";
import Spectrum from "./Spectrum";

type Status =
  | { kind: "idle" }
  | { kind: "working"; message: string; since: number }
  | { kind: "ok"; message: string }
  | { kind: "err"; message: string };

export default function Console() {
  const [csd, setCsd] = useState<string>("");
  const [runId, setRunId] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunEntryWithMeta[]>([]);
  const [editOpen, setEditOpen] = useState(false);
  const [manualEditOpen, setManualEditOpen] = useState(false);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [copied, setCopied] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [seed, setSeed] = useState<string>("");
  const [activeModelLabel, setActiveModelLabel] = useState<string>("Cstore_V1.0.2/best");

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  // ——— Library ————————————————————————————————————————————————
  const refreshLibrary = useCallback(async () => {
    try {
      const list = await listRuns();
      setRuns(list.filter((r) => r.has_wav));
    } catch {
      // silent; the library panel just stays empty
    }
  }, []);

  useEffect(() => {
    refreshLibrary();
  }, [refreshLibrary]);

  // ——— Audio graph (created once, reused for every run) ————————————
  const ensureAudioGraph = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (!audioCtxRef.current) {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const ctx = new Ctx();
      const analyserNode = ctx.createAnalyser();
      analyserNode.fftSize = 512;
      analyserNode.smoothingTimeConstant = 0.82;
      const src = ctx.createMediaElementSource(audio);
      src.connect(analyserNode);
      analyserNode.connect(ctx.destination);
      audioCtxRef.current = ctx;
      analyserRef.current = analyserNode;
      sourceRef.current = src;
      setAnalyser(analyserNode);
    }
  }, []);

  // ——— Working-state stopwatch (so the UI shows real elapsed time) —————
  useEffect(() => {
    if (status.kind !== "working") {
      setElapsed(0);
      return;
    }
    const t = window.setInterval(() => {
      setElapsed((Date.now() - status.since) / 1000);
    }, 100);
    return () => window.clearInterval(t);
  }, [status]);

  // ——— Core actions ———————————————————————————————————————————
  const runGenerate = useCallback(async () => {
    const parsedSeed = seed.trim() ? Number(seed.trim()) : undefined;
    if (parsedSeed != null && (!Number.isFinite(parsedSeed) || parsedSeed < 0)) {
      setStatus({ kind: "err", message: "Seed must be a non-negative number." });
      return;
    }
    setStatus({
      kind: "working",
      message: "Drafting .csd · rendering · verifying audio",
      since: Date.now(),
    });
    setCopied(false);
    try {
      const data = await generate({
        seed: parsedSeed,
        checkpoint: activeModelLabel || undefined,
      });
      setCsd(data.csd);
      setRunId(data.run_id);
      const audio = audioRef.current;
      if (audio) {
        audio.src = data.wav_url;
        audio.load();
      }
      setStatus({
        kind: "ok",
        message: `Render complete · run ${data.run_id.slice(-8)}`,
      });
      refreshLibrary();
    } catch (e) {
      const message = e instanceof Error ? e.message : "Generation failed";
      setStatus({ kind: "err", message });
    }
  }, [refreshLibrary, seed]);

  const loadRun = useCallback(async (id: string) => {
    try {
      const text = await fetchCsd(id);
      setCsd(text);
      setRunId(id);
      const audio = audioRef.current;
      if (audio) {
        audio.src = `/generated/${encodeURIComponent(id)}/output.wav`;
        audio.load();
      }
      setStatus({ kind: "ok", message: `Loaded ${id.slice(-8)}` });
      setCopied(false);
    } catch (e) {
      setStatus({
        kind: "err",
        message: e instanceof Error ? e.message : "Load failed",
      });
    }
  }, []);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || !audio.src) return;
    ensureAudioGraph();
    audioCtxRef.current?.resume();
    if (audio.paused) audio.play();
    else audio.pause();
  }, [ensureAudioGraph]);

  const copyCsd = useCallback(async () => {
    if (!csd) return;
    try {
      await navigator.clipboard.writeText(csd);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setStatus({
        kind: "err",
        message: "Clipboard blocked — select the source and copy manually.",
      });
    }
  }, [csd]);

  // ——— Keyboard shortcuts — only when focus isn't in a text input —————
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "g" || e.key === "G") {
        e.preventDefault();
        runGenerate();
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        copyCsd();
      } else if ((e.key === "e" || e.key === "E") && runId) {
        e.preventDefault();
        setEditOpen(true);
      } else if ((e.key === "r" || e.key === "R") && csd) {
        e.preventDefault();
        setManualEditOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runGenerate, togglePlay, copyCsd, runId, csd]);

  const handleEdited = useCallback(
    (result: EditResponse) => {
      setCsd(result.csd);
      setRunId(result.run_id);
      const audio = audioRef.current;
      if (audio) {
        audio.src = result.wav_url;
        audio.load();
      }
      setStatus({
        kind: "ok",
        message: `Edited via ${result.provider} · ${result.model} · run ${result.run_id.slice(
          -8
        )}`,
      });
      setEditOpen(false);
      refreshLibrary();
    },
    [refreshLibrary]
  );

  // Star toggle for the library. Optimistically flips the flag in local
  // state so the star animates instantly, then lets the server response
  // reconcile on the next refresh. Any backend error rolls the local
  // change back and surfaces a message in the status strip.
  const toggleFavorite = useCallback(
    async (id: string) => {
      const current = runs.find((r) => r.run_id === id);
      const next = !current?.meta?.favorite;
      setRuns((prev) =>
        prev.map((r) =>
          r.run_id === id
            ? { ...r, meta: { ...(r.meta ?? {}), favorite: next } }
            : r
        )
      );
      try {
        await setFavorite(id, next);
        refreshLibrary();
      } catch (e) {
        setRuns((prev) =>
          prev.map((r) =>
            r.run_id === id
              ? { ...r, meta: { ...(r.meta ?? {}), favorite: !next } }
              : r
          )
        );
        setStatus({
          kind: "err",
          message: e instanceof Error ? e.message : "Could not save favourite",
        });
      }
    },
    [runs, refreshLibrary]
  );

  const handleRendered = useCallback(
    (result: RenderResponse) => {
      setCsd(result.csd);
      setRunId(result.run_id);
      const audio = audioRef.current;
      if (audio) {
        audio.src = result.wav_url;
        audio.load();
      }
      setStatus({
        kind: "ok",
        message: `Manual edit rendered · run ${result.run_id.slice(-8)}`,
      });
      setManualEditOpen(false);
      refreshLibrary();
    },
    [refreshLibrary]
  );

  const working = status.kind === "working";
  const runMeta = runId ? formatRunId(runId) : null;

  // Partition the library into favourites + recent. Keeps the backend's
  // reverse-chronological order inside each bucket so the most recent
  // favourite appears on top. Cheap enough to compute on every render
  // (the list is capped to 50 + starred runs).
  const favoriteRuns = runs.filter((r) => r.meta?.favorite);
  const recentRuns = runs.filter((r) => !r.meta?.favorite);

  const handleModelChange = useCallback((active: string, selection: string | null) => {
    setActiveModelLabel(selection ?? active);
    setStatus({ kind: "ok", message: `Model loaded · ${selection ?? active}` });
  }, []);

  const handleModelError = useCallback((msg: string) => {
    setStatus({ kind: "err", message: msg });
  }, []);

  const handleModelLoading = useCallback(
    (busy: boolean, label: string | null) => {
      if (busy) {
        setStatus({
          kind: "working",
          message: `Loading model · ${label ?? ""}`.trim(),
          since: Date.now(),
        });
      }
    },
    []
  );

  return (
    <section className="mt-6">
      {/* ———— Status strip — status messages live at the top of the console
              rather than in a toast, matching the DAW aesthetic. ———— */}
      <div className="flex items-center justify-between border border-ink bg-paper-2 px-4 py-2 text-[12px]">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className={
              "inline-block h-2 w-2 " +
              (status.kind === "working"
                ? "bg-ink-red"
                : status.kind === "ok"
                ? "bg-signal"
                : status.kind === "err"
                ? "bg-ink-red"
                : "bg-ink-muted")
            }
          />
          <span className="mono tabular text-ink">
            {status.kind === "idle"
              ? "READY · awaiting input"
              : status.kind === "working"
              ? `${status.message} · ${elapsed.toFixed(1)}s`
              : status.kind === "ok"
              ? status.message
              : `ERROR · ${status.message}`}
          </span>
        </div>
        <div className="hidden items-center gap-2 md:flex">
          <span className="eyebrow">shortcuts</span>
          <span className="kbd">G</span>
          <span className="mono text-[11px] text-ink-muted">generate</span>
          <span className="kbd">␣</span>
          <span className="mono text-[11px] text-ink-muted">play</span>
          <span className="kbd">C</span>
          <span className="mono text-[11px] text-ink-muted">copy</span>
          <span className="kbd">E</span>
          <span className="mono text-[11px] text-ink-muted">edit</span>
          <span className="kbd">R</span>
          <span className="mono text-[11px] text-ink-muted">rewrite</span>
        </div>
      </div>

      {/* ———— Model selector ———————————————————————————————————— */}
      <ModelPicker
        onActiveChange={handleModelChange}
        onError={handleModelError}
        onLoadingChange={handleModelLoading}
      />

      {/* ———— Main console grid —————————————————————————————————— */}
      <div className="mt-4 grid grid-cols-12 gap-4">
        {/* ——— Left: library / file rail ———
            Sticky on md+ so the rail stays visible while the user scrolls
            through the source + terminal on the right. `self-start` keeps
            the aside from stretching to the grid row height, which is what
            lets `sticky top-…` actually pin. */}
        <aside className="col-span-12 border border-ink bg-paper-2 md:sticky md:top-4 md:col-span-3 md:self-start">
          <div className="flex items-center justify-between border-b border-ink px-3 py-2">
            <span className="eyebrow-ink">Library</span>
            <button
              onClick={refreshLibrary}
              className="mono text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink-red"
            >
              refresh
            </button>
          </div>
          {/* Split the library into two sections: Favourites (starred timbres
              the user wants pinned) and Recent. The scroll wraps BOTH so the
              user sees a continuous list, but the section headers let them
              jump visually. Favourites sort most-recent-first inside their
              group; Recent keeps the server's reverse-chronological order. */}
          <div className="scroll-ink max-h-[520px] overflow-y-auto md:max-h-[calc(100vh-7rem)]">
            {runs.length === 0 ? (
              <p className="mono px-3 py-4 text-[11px] text-ink-muted">
                No renders yet. Press{" "}
                <span className="kbd">G</span> to draft one.
              </p>
            ) : (
              <>
                {/* Favourites sub-section — only rendered when there's at
                    least one. Uses a slightly warmer row background so the
                    eye can tell the drawers apart at a glance. */}
                {favoriteRuns.length > 0 && (
                  <div>
                    <div className="flex items-center justify-between border-b border-rule-2 bg-paper-3/60 px-3 py-1.5">
                      <span className="eyebrow">favourites</span>
                      <span className="mono tabular text-[10px] text-ink-muted">
                        {favoriteRuns.length}
                      </span>
                    </div>
                    <ol>
                      {favoriteRuns.map((r, i) => (
                        <LibraryRow
                          key={r.run_id}
                          run={r}
                          index={favoriteRuns.length - i}
                          active={r.run_id === runId}
                          onLoad={loadRun}
                          onToggleFavorite={toggleFavorite}
                        />
                      ))}
                    </ol>
                  </div>
                )}

                {/* Recent sub-section — everything that isn't starred. */}
                <div>
                  {favoriteRuns.length > 0 && (
                    <div className="flex items-center justify-between border-b border-rule-2 bg-paper-2 px-3 py-1.5">
                      <span className="eyebrow">recent</span>
                      <span className="mono tabular text-[10px] text-ink-muted">
                        {recentRuns.length}
                      </span>
                    </div>
                  )}
                  {recentRuns.length === 0 ? (
                    <p className="mono px-3 py-3 text-[11px] text-ink-muted">
                      Everything here is a favourite — generate a new
                      timbre with <span className="kbd">G</span>.
                    </p>
                  ) : (
                    <ol>
                      {recentRuns.map((r, i) => (
                        <LibraryRow
                          key={r.run_id}
                          run={r}
                          index={recentRuns.length - i}
                          active={r.run_id === runId}
                          onLoad={loadRun}
                          onToggleFavorite={toggleFavorite}
                        />
                      ))}
                    </ol>
                  )}
                </div>
              </>
            )}
          </div>
        </aside>

        {/* ——— Center: studio panel (spectrum + transport + source) ——— */}
        <div className="col-span-12 md:col-span-9">
          <div className="grid grid-cols-12 gap-4">
            {/* Spectrum + transport */}
            <div className="col-span-12 border border-ink bg-paper-2 lg:col-span-7">
              <div className="flex items-center justify-between border-b border-ink px-3 py-2">
                <span className="eyebrow-ink">Monitor · FFT / Scope</span>
                <span className="mono tabular text-[11px] text-ink-muted">
                  512-point · Hann · 44.1k
                </span>
              </div>

              <div className="relative p-3">
                {/* 10% tick ruler above the canvas */}
                <div className="tick-rule mb-1 h-2 opacity-50" />
                <Spectrum
                  analyser={analyser}
                  playing={playing}
                  placeholder={
                    working
                      ? "rendering…"
                      : csd
                      ? "press play to inspect"
                      : "generate to begin"
                  }
                />

                {/* crosshatch overlay while working */}
                {working && (
                  <div className="hatch pointer-events-none absolute inset-3 top-5" />
                )}

                {/* Transport row */}
                <div className="mt-3 flex items-center gap-3">
                  <button
                    onClick={runGenerate}
                    disabled={working}
                    className="flex items-center gap-2 border border-ink bg-ink px-4 py-2 text-paper disabled:opacity-60"
                  >
                    <span aria-hidden>●</span>
                    <span className="mono text-[12px] uppercase tracking-widest">
                      {working ? "Rendering" : "Generate"}
                    </span>
                    <span className="kbd !border-paper !bg-ink !text-paper">
                      G
                    </span>
                  </button>

                  <button
                    onClick={togglePlay}
                    disabled={!csd}
                    className="flex items-center gap-2 border border-ink bg-paper px-3 py-2 disabled:opacity-40"
                  >
                    <span aria-hidden>{playing ? "❚❚" : "▶"}</span>
                    <span className="mono text-[12px] uppercase tracking-widest">
                      {playing ? "Pause" : "Play"}
                    </span>
                  </button>

                  {/* Seed input — lets the user pin a run. */}
                  <label className="ml-auto flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">seed</span>
                    <input
                      type="number"
                      inputMode="numeric"
                      min={0}
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                      placeholder="auto"
                      className="mono w-24 bg-transparent text-[12px] outline-none placeholder:text-ink-muted"
                    />
                  </label>
                </div>

                {/* Hidden audio element, controlled by the transport */}
                <audio
                  ref={audioRef}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onEnded={() => setPlaying(false)}
                  preload="metadata"
                  hidden
                />
              </div>
            </div>

            {/* Run metadata card */}
            <div className="col-span-12 border border-ink bg-paper-2 lg:col-span-5">
              <div className="flex items-center justify-between border-b border-ink px-3 py-2">
                <span className="eyebrow-ink">Run</span>
                <span className="mono text-[11px] text-ink-muted">
                  {runMeta ? `#${runMeta.hash}` : "—"}
                </span>
              </div>
              <dl className="divide-y divide-rule-2">
                <Row label="date" value={runMeta ? runMeta.date : "—"} />
                <Row label="time" value={runMeta ? runMeta.time : "—"} />
                <Row
                  label="checkpoint"
                  value={activeModelLabel || "—"}
                  hint={
                    /Cstore_V1\.0\.1\/best/.test(activeModelLabel)
                      ? "73% · 56% · 54%"
                      : undefined
                  }
                />
                <Row label="sampling" value="T=0.8 · top-p=0.9 · max 400 tok" />
                <Row
                  label="render"
                  value={csd ? "csound ✓ · RMS > 1e-4 ✓" : "—"}
                />
              </dl>
              <div className="flex items-center gap-2 border-t border-ink px-3 py-2">
                <a
                  href={
                    runId
                      ? `/generated/${encodeURIComponent(runId)}/output.csd`
                      : "#"
                  }
                  download
                  aria-disabled={!runId}
                  className={
                    "mono text-[11px] uppercase tracking-widest underline decoration-ink-red underline-offset-4 " +
                    (runId ? "text-ink" : "pointer-events-none text-ink-muted")
                  }
                >
                  download .csd
                </a>
                <span className="text-ink-muted">·</span>
                <a
                  href={
                    runId
                      ? `/generated/${encodeURIComponent(runId)}/output.wav`
                      : "#"
                  }
                  download
                  aria-disabled={!runId}
                  className={
                    "mono text-[11px] uppercase tracking-widest underline decoration-ink-red underline-offset-4 " +
                    (runId ? "text-ink" : "pointer-events-none text-ink-muted")
                  }
                >
                  download .wav
                </a>
                <button
                  type="button"
                  onClick={() => setManualEditOpen(true)}
                  disabled={!csd}
                  className="ml-auto mono flex items-center gap-1 text-[11px] uppercase tracking-widest text-ink hover:text-ink-red disabled:pointer-events-none disabled:text-ink-muted"
                >
                  edit manually
                  <span className="kbd">R</span>
                </button>
                <button
                  type="button"
                  onClick={() => setEditOpen(true)}
                  disabled={!runId}
                  className="mono flex items-center gap-1 text-[11px] uppercase tracking-widest text-ink hover:text-ink-red disabled:pointer-events-none disabled:text-ink-muted"
                >
                  edit with llm
                  <span className="kbd">E</span>
                </button>
                <a
                  href="https://ide.csound.com/editor/OE3qtlvC0RKjq47vPDdx"
                  target="_blank"
                  rel="noreferrer"
                  className="mono text-[11px] uppercase tracking-widest text-ink-red hover:text-ink-red-2"
                >
                  open in csound ide →
                </a>
              </div>
              {/* Show edit provenance when this run was created by an LLM
                  edit OR a hand edit. Both carry `derived_from` so the user
                  can trace what the source was. */}
              {(() => {
                const meta = runs.find((r) => r.run_id === runId)?.meta;
                if (!meta) return null;
                if (meta.kind === "edit") {
                  return (
                    <div className="border-t border-rule-2 bg-paper-3/40 px-3 py-2">
                      <div className="eyebrow">edited from</div>
                      <div className="mono tabular mt-1 text-[11px]">
                        #{meta.derived_from?.slice(-8)} · via {meta.provider}{" "}
                        · {meta.model}
                      </div>
                      {meta.instruction && (
                        <div className="mono mt-1 text-[11px] italic text-ink-muted">
                          “{meta.instruction}”
                        </div>
                      )}
                    </div>
                  );
                }
                if (meta.kind === "manual_edit") {
                  return (
                    <div className="border-t border-rule-2 bg-paper-3/40 px-3 py-2">
                      <div className="eyebrow">hand-edited from</div>
                      <div className="mono tabular mt-1 text-[11px]">
                        {meta.derived_from
                          ? `#${meta.derived_from.slice(-8)} · rendered locally`
                          : "ad-hoc source · rendered locally"}
                      </div>
                    </div>
                  );
                }
                return null;
              })()}
            </div>

            {/* CSD source inspector */}
            <div className="col-span-12 border border-ink bg-paper-2">
              <div className="flex items-center justify-between border-b border-ink px-3 py-2">
                <span className="eyebrow-ink">Source · output.csd</span>
                <div className="flex items-center gap-3">
                  <span className="mono tabular text-[11px] text-ink-muted">
                    {csd ? `${csd.split("\n").length} lines · ${csd.length} chars` : "—"}
                  </span>
                  <button
                    onClick={copyCsd}
                    disabled={!csd}
                    className="mono flex items-center gap-1 border border-ink bg-paper px-2 py-1 text-[11px] uppercase tracking-widest disabled:opacity-40"
                  >
                    {copied ? "copied ✓" : "copy"}
                    {!copied && <span className="kbd">C</span>}
                  </button>
                </div>
              </div>
              {/* The scroll container wraps BOTH columns so line numbers
                  scroll vertically together with the code. The gutter uses
                  `sticky left-0` so horizontal scroll on long lines keeps
                  the numbers pinned instead of sliding off-screen. */}
              <div className="scroll-ink grid max-h-[420px] grid-cols-[3rem_1fr] overflow-auto">
                <pre className="mono tabular sticky left-0 z-10 select-none whitespace-pre border-r border-rule-2 bg-paper-3/40 py-3 text-right text-[11px] leading-[1.55] text-ink-muted">
                  {csd
                    ? csd
                        .split("\n")
                        .map((_, i) => String(i + 1).padStart(3, " "))
                        .join("\n")
                    : "  1"}
                </pre>
                <pre className="mono whitespace-pre p-3 text-[12px] leading-[1.55] text-ink">
                  {csd || "# Press G to draft a .csd — or pick a render from the library."}
                </pre>
              </div>
            </div>

            {/* Live csound terminal — mirrors what the Python sidecar and the
                csound binary are printing right now. Useful for debugging a
                silent render or watching the watchdog retry loop. */}
            <CsoundTerminal />
          </div>
        </div>
      </div>

      {/* ———— Edit-with-LLM dialog (mounted but gated by `open`) ———— */}
      <EditDialog
        open={editOpen}
        runId={runId}
        onClose={() => setEditOpen(false)}
        onEdited={handleEdited}
      />

      {/* ———— Hand-edit dialog — edit the raw .csd and re-render with Csound
              (no model is invoked). Seeded from whatever `csd` is currently
              loaded into the console so the user always starts from the
              source they can see in the inspector. ———— */}
      <ManualEditDialog
        open={manualEditOpen}
        runId={runId}
        initialCsd={csd}
        onClose={() => setManualEditOpen(false)}
        onRendered={handleRendered}
      />
    </section>
  );
}

function Row({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between px-3 py-2">
      <dt className="eyebrow">{label}</dt>
      <dd className="mono tabular text-right text-[12px]">
        {value}
        {hint && (
          <span className="ml-2 text-[11px] text-ink-muted">{hint}</span>
        )}
      </dd>
    </div>
  );
}

/**
 * One row of the library drawer. Extracted so both the Favourites and
 * Recent sections render identically — the only differences are the
 * positional index (numbering is per-section) and which list they live in.
 *
 * The star button stops event propagation so clicking it toggles favourite
 * status WITHOUT also triggering the row's `onLoad` handler.
 */
function LibraryRow({
  run,
  index,
  active,
  onLoad,
  onToggleFavorite,
}: {
  run: RunEntryWithMeta;
  index: number;
  active: boolean;
  onLoad: (runId: string) => void;
  onToggleFavorite: (runId: string) => void;
}) {
  const meta = formatRunId(run.run_id);
  const starred = Boolean(run.meta?.favorite);
  return (
    <li
      className={
        "group flex cursor-pointer items-baseline gap-3 border-b border-rule-2 px-3 py-2 " +
        (active ? "bg-paper-3" : "hover:bg-paper-3/60")
      }
      onClick={() => onLoad(run.run_id)}
    >
      <span className="mono tabular w-6 text-[10px] text-ink-muted">
        {String(index).padStart(3, "0")}
      </span>
      <div className="flex-1">
        <div className="mono tabular flex items-center gap-2 text-[12px]">
          {meta.date} · {meta.time}
          {run.meta?.kind === "edit" && (
            <span
              className="mono border border-ink-red px-1 text-[9px] uppercase tracking-widest text-ink-red"
              title={run.meta.instruction}
            >
              edit
            </span>
          )}
          {run.meta?.kind === "manual_edit" && (
            <span
              className="mono border border-ink px-1 text-[9px] uppercase tracking-widest text-ink"
              title="hand-edited source · rendered locally"
            >
              manual
            </span>
          )}
        </div>
        <div className="mono text-[10px] text-ink-muted">
          #{meta.hash} ·{" "}
          {run.meta?.kind === "edit"
            ? `${run.meta.provider} · ${run.meta.model}`
            : run.meta?.kind === "manual_edit"
            ? "hand-edited · csound"
            : ".csd + .wav"}
        </div>
      </div>
      <button
        type="button"
        aria-label={starred ? "Remove from favourites" : "Add to favourites"}
        aria-pressed={starred}
        title={starred ? "Remove from favourites" : "Save as favourite"}
        onClick={(e) => {
          e.stopPropagation();
          onToggleFavorite(run.run_id);
        }}
        className={
          "mono flex h-5 w-5 items-center justify-center text-[13px] leading-none transition-colors " +
          (starred
            ? "text-ink-red hover:text-ink-red-2"
            : "text-ink-muted opacity-0 group-hover:opacity-100 hover:text-ink")
        }
      >
        {starred ? "★" : "☆"}
      </button>
      <span
        className={
          "mono w-3 text-[10px] " +
          (active ? "text-ink-red" : "text-ink-muted group-hover:text-ink")
        }
      >
        {active ? "●" : "→"}
      </span>
    </li>
  );
}
