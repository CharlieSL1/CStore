"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteRun,
  fetchCsd,
  fetchServiceHealth,
  formatRunId,
  generate,
  generateChildren,
  generateChildrenLlm,
  generateStarters,
  listRuns,
  setFavorite,
  type EditResponse,
  type LlmCost,
  type LlmProvider,
  type LlmUsage,
  type NoteMode,
  type NoteRangeMode,
  type RenderResponse,
  type RunEntryWithMeta,
  type StarterChild,
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

type ChildBatchKind = "starter" | "child-classic" | "child-llm";
type ChildBatch = {
  id: string;
  kind: ChildBatchKind;
  motherRunId: string | null;
  mode: string;
  variation: number | null;
  createdAt: number;
  children: StarterChild[];
};

type SessionLlmTotals = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_usd: number;
  has_prompt_tokens: boolean;
  has_completion_tokens: boolean;
  has_total_tokens: boolean;
  has_estimated_usd: boolean;
};

function modeLabel(mode?: string): string {
  if (!mode) return "single";
  return mode === "scale" ? "arpeggio" : mode;
}

const LLM_CHILD_PROVIDERS: Array<{ id: LlmProvider; label: string }> = [
  { id: "qwen", label: "Qwen (local/free)" },
  { id: "pollinations", label: "Pollinations (free)" },
  { id: "openrouter", label: "OpenRouter (free w/ key)" },
  { id: "openai", label: "OpenAI (key)" },
  { id: "anthropic", label: "Anthropic (key)" },
  { id: "gemini", label: "Gemini (key)" },
];

const DEFAULT_CHILD_LLM_MODEL: Record<LlmProvider, string> = {
  qwen: "qwen3.6:35b-a3b-coding-mxfp8",
  pollinations: "openai",
  openrouter: "openrouter/free",
  openai: "gpt-5.4",
  anthropic: "claude-opus-4-7",
  gemini: "gemini-3.1-pro",
};

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
  const [starterCount, setStarterCount] = useState<string>("1");
  const [maxTokens, setMaxTokens] = useState<string>("400");
  const [noteDuration, setNoteDuration] = useState<string>("1.5");
  const [noteMode, setNoteMode] = useState<NoteMode>("single");
  const [noteRangeMode, setNoteRangeMode] = useState<NoteRangeMode>("auto");
  const [noteLowMidi, setNoteLowMidi] = useState<string>("48");
  const [noteHighMidi, setNoteHighMidi] = useState<string>("72");
  const [childGenMode, setChildGenMode] = useState<"classic" | "llm">("llm");
  const [childLlmProvider, setChildLlmProvider] = useState<LlmProvider>("qwen");
  const [childLlmModel, setChildLlmModel] = useState<string>("qwen3.6:35b-a3b-coding-mxfp8");
  const [childLlmThink, setChildLlmThink] = useState<boolean>(true);
  const [childLlmVariationTemp, setChildLlmVariationTemp] = useState<string>("0.35");
  const [reverbMix, setReverbMix] = useState<string>("0.18");
  const [childBatches, setChildBatches] = useState<ChildBatch[]>([]);
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null);
  const [activeModelLabel, setActiveModelLabel] = useState<string>("Cstore_V1.0.2/best");
  const [serviceReady, setServiceReady] = useState<boolean>(true);
  const [lastLlmUsage, setLastLlmUsage] = useState<LlmUsage | null>(null);
  const [lastLlmCost, setLastLlmCost] = useState<LlmCost | null>(null);
  const [sessionLlmTotals, setSessionLlmTotals] = useState<SessionLlmTotals>({
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    estimated_usd: 0,
    has_prompt_tokens: false,
    has_completion_tokens: false,
    has_total_tokens: false,
    has_estimated_usd: false,
  });

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const compressorRef = useRef<DynamicsCompressorNode | null>(null);
  const reverbDryRef = useRef<GainNode | null>(null);
  const reverbWetRef = useRef<GainNode | null>(null);
  const convolverRef = useRef<ConvolverNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  useEffect(() => {
    setChildLlmModel(DEFAULT_CHILD_LLM_MODEL[childLlmProvider]);
  }, [childLlmProvider]);

  // ——— Library ————————————————————————————————————————————————
  const refreshLibrary = useCallback(async (quiet = false) => {
    try {
      await fetchServiceHealth();
      const list = await listRuns();
      setRuns(list.filter((r) => r.has_wav));
      setServiceReady(true);
      return true;
    } catch (e) {
      setServiceReady(false);
      if (!quiet) {
        setStatus({
          kind: "err",
          message:
            e instanceof Error
              ? `${e.message} · refresh backend and try again`
              : "Backend unavailable · refresh backend and try again",
        });
      }
      return false;
    }
  }, []);

  const upsertLibraryRuns = useCallback((nextRuns: RunEntryWithMeta[]) => {
    setRuns((prev) => {
      const map = new Map<string, RunEntryWithMeta>();
      for (const run of prev) map.set(run.run_id, run);
      for (const run of nextRuns) map.set(run.run_id, run);
      return Array.from(map.values()).sort((a, b) =>
        b.run_id.localeCompare(a.run_id)
      );
    });
  }, []);

  const appendChildBatch = useCallback((batch: Omit<ChildBatch, "id" | "createdAt">) => {
    const id = `batch_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
    const createdAt = Date.now();
    setChildBatches((prev) => [...prev, { ...batch, id, createdAt }]);
    setActiveBatchId(id);
  }, []);

  const removeChildBatch = useCallback((id: string) => {
    setChildBatches((prev) => {
      const next = prev.filter((b) => b.id !== id);
      if (activeBatchId === id) {
        setActiveBatchId(next.length ? next[next.length - 1].id : null);
      }
      return next;
    });
  }, [activeBatchId]);

  const clearChildBatches = useCallback(() => {
    setChildBatches([]);
    setActiveBatchId(null);
  }, []);

  const updateLlmMeter = useCallback((usage?: LlmUsage, cost?: LlmCost) => {
    setLastLlmUsage(usage ?? null);
    setLastLlmCost(cost ?? null);
    setSessionLlmTotals((prev) => {
      const next = { ...prev };
      if (typeof usage?.prompt_tokens === "number" && Number.isFinite(usage.prompt_tokens)) {
        next.prompt_tokens += usage.prompt_tokens;
        next.has_prompt_tokens = true;
      }
      if (
        typeof usage?.completion_tokens === "number" &&
        Number.isFinite(usage.completion_tokens)
      ) {
        next.completion_tokens += usage.completion_tokens;
        next.has_completion_tokens = true;
      }
      if (typeof usage?.total_tokens === "number" && Number.isFinite(usage.total_tokens)) {
        next.total_tokens += usage.total_tokens;
        next.has_total_tokens = true;
      }
      if (typeof cost?.estimated_usd === "number" && Number.isFinite(cost.estimated_usd)) {
        next.estimated_usd += cost.estimated_usd;
        next.has_estimated_usd = true;
      }
      return next;
    });
  }, []);

  useEffect(() => {
    refreshLibrary(true);
  }, [refreshLibrary]);

  // Keep the library live-updating so newly rendered runs appear even if the
  // user doesn't click the manual refresh button.
  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshLibrary(true);
    }, 4000);
    return () => window.clearInterval(id);
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
      const gainNode = ctx.createGain();
      gainNode.gain.value = 0.75;
      const compressorNode = ctx.createDynamicsCompressor();
      compressorNode.threshold.value = -14;
      compressorNode.knee.value = 18;
      compressorNode.ratio.value = 8;
      compressorNode.attack.value = 0.003;
      compressorNode.release.value = 0.18;
      const dryNode = ctx.createGain();
      const wetNode = ctx.createGain();
      const convolver = ctx.createConvolver();
      const src = ctx.createMediaElementSource(audio);

      const impulseSeconds = 1.8;
      const impulseLength = Math.floor(ctx.sampleRate * impulseSeconds);
      const impulse = ctx.createBuffer(2, impulseLength, ctx.sampleRate);
      for (let channel = 0; channel < impulse.numberOfChannels; channel += 1) {
        const data = impulse.getChannelData(channel);
        for (let i = 0; i < impulseLength; i += 1) {
          const t = i / impulseLength;
          data[i] = (Math.random() * 2 - 1) * (1 - t) ** 2.1;
        }
      }
      convolver.buffer = impulse;

      src.connect(gainNode);
      gainNode.connect(compressorNode);
      compressorNode.connect(dryNode);
      compressorNode.connect(convolver);
      convolver.connect(wetNode);
      dryNode.connect(analyserNode);
      wetNode.connect(analyserNode);
      analyserNode.connect(ctx.destination);
      audioCtxRef.current = ctx;
      analyserRef.current = analyserNode;
      gainRef.current = gainNode;
      compressorRef.current = compressorNode;
      reverbDryRef.current = dryNode;
      reverbWetRef.current = wetNode;
      convolverRef.current = convolver;
      sourceRef.current = src;
      setAnalyser(analyserNode);
    }
  }, []);

  useEffect(() => {
    const wet = Number(reverbMix);
    if (!Number.isFinite(wet)) return;
    const clampedWet = Math.max(0, Math.min(1, wet));
    const dry = 1 - clampedWet;
    if (reverbDryRef.current) reverbDryRef.current.gain.value = dry;
    if (reverbWetRef.current) reverbWetRef.current.gain.value = clampedWet;
  }, [reverbMix]);

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
  const parseSeed = useCallback(() => {
    const parsedSeed = seed.trim() ? Number(seed.trim()) : undefined;
    if (parsedSeed != null && (!Number.isFinite(parsedSeed) || parsedSeed < 0)) {
      throw new Error("Seed must be a non-negative number.");
    }
    return parsedSeed;
  }, [seed]);

  const parseBoundedNumber = useCallback(
    (value: string, label: string, min: number, max: number) => {
      const parsed = Number(value);
      if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
        throw new Error(`${label} must be an integer from ${min} to ${max}.`);
      }
      return parsed;
    },
    []
  );

  const parseBoundedFloat = useCallback(
    (value: string, label: string, min: number, max: number) => {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
        throw new Error(`${label} must be from ${min} to ${max}.`);
      }
      return parsed;
    },
    []
  );

  const runGenerate = useCallback(async () => {
    let parsedSeed: number | undefined;
    let parsedMaxTokens: number;
    let parsedNoteDuration: number;
    let parsedLowMidi: number;
    let parsedHighMidi: number;
    try {
      parsedSeed = parseSeed();
      parsedMaxTokens = parseBoundedNumber(maxTokens, "Max tokens", 100, 500);
      parsedNoteDuration = parseBoundedFloat(noteDuration, "Note duration", 0.1, 10);
      parsedLowMidi = parseBoundedNumber(noteLowMidi, "Low note", 24, 108);
      parsedHighMidi = parseBoundedNumber(noteHighMidi, "High note", 24, 108);
    } catch (e) {
      setStatus({ kind: "err", message: e instanceof Error ? e.message : "Invalid input" });
      return;
    }
    setStatus({
      kind: "working",
      message: `Generating render · max ${parsedMaxTokens} tok · verifying audio`,
      since: Date.now(),
    });
    setCopied(false);
    try {
      const data = await generate({
        seed: parsedSeed,
        checkpoint: activeModelLabel || undefined,
        max_tokens: parsedMaxTokens,
        note_duration: parsedNoteDuration,
        note_mode: noteMode,
        note_range_mode: noteRangeMode,
        note_low_midi: Math.min(parsedLowMidi, parsedHighMidi),
        note_high_midi: Math.max(parsedLowMidi, parsedHighMidi),
      });
      setCsd(data.csd);
      setRunId(data.run_id);
      upsertLibraryRuns([
        { run_id: data.run_id, has_csd: true, has_wav: true },
      ]);
      const audio = audioRef.current;
      if (audio) {
        audio.src = data.wav_url;
        audio.load();
      }
      setStatus({
        kind: "ok",
        message: `Render complete · run ${data.run_id.slice(-8)}`,
      });
      await refreshLibrary(true);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Generation failed";
      setStatus({ kind: "err", message });
    }
  }, [
    activeModelLabel,
    appendChildBatch,
    maxTokens,
    noteDuration,
    noteHighMidi,
    noteLowMidi,
    noteMode,
    noteRangeMode,
    parseBoundedFloat,
    parseBoundedNumber,
    parseSeed,
    refreshLibrary,
    upsertLibraryRuns,
  ]);

  const runGenerateStarters = useCallback(async () => {
    let parsedSeed: number | undefined;
    let parsedCount: number;
    let parsedMaxTokens: number;
    let parsedNoteDuration: number;
    let parsedLowMidi: number;
    let parsedHighMidi: number;
    try {
      parsedSeed = parseSeed();
      parsedCount = parseBoundedNumber(starterCount, "Outputs", 1, 12);
      parsedMaxTokens = parseBoundedNumber(maxTokens, "Max tokens", 100, 500);
      parsedNoteDuration = parseBoundedFloat(noteDuration, "Note duration", 0.1, 10);
      parsedLowMidi = parseBoundedNumber(noteLowMidi, "Low note", 24, 108);
      parsedHighMidi = parseBoundedNumber(noteHighMidi, "High note", 24, 108);
    } catch (e) {
      setStatus({ kind: "err", message: e instanceof Error ? e.message : "Invalid input" });
      return;
    }
    setStatus({
      kind: "working",
      message:
        `Generating ${parsedCount} starters · ${noteMode} · ${parsedNoteDuration}s` +
        ` · range ${noteRangeMode === "auto" ? "auto-2oct" : `${Math.min(parsedLowMidi, parsedHighMidi)}-${Math.max(parsedLowMidi, parsedHighMidi)} midi`}` +
        ` · max ${parsedMaxTokens} tok`,
      since: Date.now(),
    });
    setCopied(false);
    try {
      const data = await generateStarters({
        count: parsedCount,
        seed: parsedSeed,
        checkpoint: activeModelLabel || undefined,
        max_tokens: parsedMaxTokens,
        note_duration: parsedNoteDuration,
        note_mode: noteMode,
        note_range_mode: noteRangeMode,
        note_low_midi: Math.min(parsedLowMidi, parsedHighMidi),
        note_high_midi: Math.max(parsedLowMidi, parsedHighMidi),
      });
      appendChildBatch({
        kind: "starter",
        motherRunId: null,
        mode: modeLabel(data.note_mode ?? noteMode),
        variation: null,
        children: data.starters,
      });
      upsertLibraryRuns(
        data.starters.map((child) => ({
          run_id: child.run_id,
          has_csd: true,
          has_wav: true,
          meta: {
            kind: "starter",
            starter_type: child.starter_type,
            note_mode: child.note_mode,
            child_index: child.child_index,
            duration_sec: child.duration_sec,
          },
        }))
      );
      const first = data.starters[0];
      if (first) {
        setCsd(first.csd);
        setRunId(first.run_id);
        const audio = audioRef.current;
        if (audio) {
          audio.src = first.wav_url;
          audio.load();
        }
      }
      setStatus({
        kind: "ok",
        message: `Generated ${data.count} starters · ${modeLabel(data.note_mode ?? noteMode)} · ${data.duration_sec}s`,
      });
      await refreshLibrary(true);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Starter generation failed";
      setStatus({ kind: "err", message });
    }
  }, [
    activeModelLabel,
    appendChildBatch,
    maxTokens,
    noteDuration,
    noteHighMidi,
    noteLowMidi,
    noteMode,
    noteRangeMode,
    parseBoundedFloat,
    parseBoundedNumber,
    parseSeed,
    refreshLibrary,
    starterCount,
    upsertLibraryRuns,
  ]);

  const runGenerateChildren = useCallback(async () => {
    if (!runId) {
      setStatus({
        kind: "err",
        message: "Select a source run before generating children.",
      });
      return;
    }
    const sourceRun = runs.find((r) => r.run_id === runId);
    if (!sourceRun) {
      setStatus({
        kind: "err",
        message: `Source run #${runId.slice(-8)} is not in the library. Refresh and retry.`,
      });
      return;
    }
    let parsedCount: number;
    let parsedNoteDuration: number;
    let parsedLowMidi: number;
    let parsedHighMidi: number;
    let parsedVariationTemp: number;
    try {
      parsedCount = parseBoundedNumber(starterCount, "Outputs", 1, 12);
      parsedNoteDuration = parseBoundedFloat(noteDuration, "Note duration", 0.1, 10);
      parsedLowMidi = parseBoundedNumber(noteLowMidi, "Low note", 24, 108);
      parsedHighMidi = parseBoundedNumber(noteHighMidi, "High note", 24, 108);
      parsedVariationTemp = parseBoundedFloat(
        childLlmVariationTemp,
        "LLM variation",
        0,
        1
      );
      if (childGenMode === "llm" && !childLlmModel.trim()) {
        throw new Error("LLM model is required for LLM child generation.");
      }
    } catch (e) {
      setStatus({ kind: "err", message: e instanceof Error ? e.message : "Invalid input" });
      return;
    }
    setStatus({
      kind: "working",
      message:
        childGenMode === "llm"
          ? `Creating ${parsedCount} LLM children from mother #${runId.slice(-8)}`
          : `Creating ${parsedCount} classic children from mother #${runId.slice(-8)} · ${noteMode} · range ${noteRangeMode === "auto" ? "auto-2oct" : `${Math.min(parsedLowMidi, parsedHighMidi)}-${Math.max(parsedLowMidi, parsedHighMidi)} midi`}`,
      since: Date.now(),
    });
    setCopied(false);
    try {
      const data =
        childGenMode === "llm"
          ? await generateChildrenLlm({
              source_run_id: runId,
              count: parsedCount,
              provider: childLlmProvider,
              model: childLlmModel.trim(),
              think: childLlmProvider === "qwen" ? childLlmThink : undefined,
              variation_temperature: parsedVariationTemp,
            })
          : await generateChildren({
              source_run_id: runId,
              count: parsedCount,
              note_duration: parsedNoteDuration,
              note_mode: noteMode,
              note_range_mode: noteRangeMode,
              note_low_midi: Math.min(parsedLowMidi, parsedHighMidi),
              note_high_midi: Math.max(parsedLowMidi, parsedHighMidi),
            });
      if (!data.children?.length) {
        throw new Error("No children were returned by the backend.");
      }
      const isLlmBatch = "provider" in data && "model" in data;
      const batchMode = isLlmBatch
        ? modeLabel(data.children[0]?.note_mode ?? noteMode)
        : modeLabel(data.note_mode ?? noteMode);
      appendChildBatch({
        kind: isLlmBatch ? "child-llm" : "child-classic",
        motherRunId: data.derived_from,
        mode: batchMode,
        variation:
          isLlmBatch && typeof data.variation_temperature === "number"
            ? data.variation_temperature
            : null,
        children: data.children,
      });
      upsertLibraryRuns(
        data.children.map((child) => ({
          run_id: child.run_id,
          has_csd: true,
          has_wav: true,
          meta: {
            kind: "child",
            note_mode: child.note_mode,
            starter_type: child.starter_type,
            child_index: child.child_index,
            duration_sec: child.duration_sec,
            derived_from: child.derived_from,
            ...(isLlmBatch
              ? {
                  provider: data.provider,
                  model: data.model,
                  instruction: "auto child variation",
                  llm_usage: data.usage,
                  llm_cost: data.cost,
                  llm_variation_tier: data.variation_tier,
                  llm_variation_mode: data.variation_mode,
                  llm_variation_temperature: data.variation_temperature,
                }
              : {}),
          },
        }))
      );
      const first = data.children[0];
      if (first) {
        setCsd(first.csd);
        setRunId(first.run_id);
        const audio = audioRef.current;
        if (audio) {
          audio.src = first.wav_url;
          audio.load();
        }
      }
      setStatus({
        kind: "ok",
        message:
          isLlmBatch
            ? `Generated ${data.count} LLM children · ${data.provider}/${data.model} · ${data.variation_tier ?? "var"} · from mother #${data.derived_from.slice(-8)}`
            : `Generated ${data.count} children · ${modeLabel(data.note_mode)} · range ${noteRangeMode === "auto" ? "auto-2oct" : `${Math.min(parsedLowMidi, parsedHighMidi)}-${Math.max(parsedLowMidi, parsedHighMidi)} midi`} · from mother #${data.derived_from.slice(-8)}`,
      });
      if (isLlmBatch) {
        updateLlmMeter(data.usage, data.cost);
      }
      await refreshLibrary(true);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Child generation failed";
      setStatus({ kind: "err", message });
    }
  }, [
    childGenMode,
    appendChildBatch,
    childLlmModel,
    childLlmProvider,
    childLlmThink,
    childLlmVariationTemp,
    noteDuration,
    noteHighMidi,
    noteLowMidi,
    noteMode,
    noteRangeMode,
    parseBoundedFloat,
    parseBoundedNumber,
    refreshLibrary,
    runs,
    runId,
    starterCount,
    updateLlmMeter,
    upsertLibraryRuns,
  ]);

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

  const loadStarterChild = useCallback((child: StarterChild) => {
    setCsd(child.csd);
    setRunId(child.run_id);
    const audio = audioRef.current;
    if (audio) {
      audio.src = child.wav_url;
      audio.load();
    }
    setStatus({
      kind: "ok",
      message: `Loaded ${child.derived_from ? "child" : "starter"} ${child.child_index} · ${modeLabel(child.note_mode ?? child.starter_type)}`,
    });
    setCopied(false);
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
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        runGenerateStarters();
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
  }, [runGenerate, runGenerateStarters, togglePlay, copyCsd, runId, csd]);

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
        message:
          result.cost?.estimated_usd != null
            ? `Edited via ${result.provider} · ${result.model} · $${result.cost.estimated_usd.toFixed(6)}`
            : `Edited via ${result.provider} · ${result.model} · run ${result.run_id.slice(-8)}`,
      });
      updateLlmMeter(result.usage, result.cost);
      upsertLibraryRuns([
        {
          run_id: result.run_id,
          has_csd: true,
          has_wav: true,
          meta: {
            kind: "edit",
            derived_from: result.derived_from,
            provider: result.provider,
            model: result.model,
            instruction: result.instruction,
            llm_usage: result.usage,
            llm_cost: result.cost,
          },
        },
      ]);
      setEditOpen(false);
      void refreshLibrary(true);
    },
    [refreshLibrary, updateLlmMeter, upsertLibraryRuns]
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
        await refreshLibrary(true);
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

  const handleDeleteRun = useCallback(
    async (id: string) => {
      if (!window.confirm(`Delete render #${id.slice(-8)} from the library?`)) {
        return;
      }
      try {
        await deleteRun(id);
        setRuns((prev) => prev.filter((r) => r.run_id !== id));
        setChildBatches((prev) =>
          prev
            .map((batch) => ({
              ...batch,
              children: batch.children.filter((child) => child.run_id !== id),
            }))
            .filter((batch) => batch.children.length > 0)
        );
        if (runId === id) {
          setCsd("");
          setRunId(null);
          setCopied(false);
          const audio = audioRef.current;
          if (audio) {
            audio.pause();
            audio.removeAttribute("src");
            audio.load();
          }
        }
        setStatus({ kind: "ok", message: `Deleted ${id.slice(-8)}` });
        await refreshLibrary(true);
      } catch (e) {
        setStatus({
          kind: "err",
          message: e instanceof Error ? e.message : "Delete failed",
        });
      }
    },
    [refreshLibrary, runId]
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
      upsertLibraryRuns([
        {
          run_id: result.run_id,
          has_csd: true,
          has_wav: true,
          meta: {
            kind: "manual_edit",
            derived_from: result.derived_from ?? undefined,
          },
        },
      ]);
      setManualEditOpen(false);
      void refreshLibrary(true);
    },
    [refreshLibrary, upsertLibraryRuns]
  );

  const working = status.kind === "working";
  const runMeta = runId ? formatRunId(runId) : null;
  const activeBatch =
    childBatches.find((batch) => batch.id === activeBatchId) ??
    childBatches[childBatches.length - 1] ??
    null;
  const generatedChildrenMotherId = activeBatch?.motherRunId ?? null;

  useEffect(() => {
    if (!childBatches.length) {
      if (activeBatchId !== null) setActiveBatchId(null);
      return;
    }
    if (!activeBatchId || !childBatches.some((batch) => batch.id === activeBatchId)) {
      setActiveBatchId(childBatches[childBatches.length - 1].id);
    }
  }, [activeBatchId, childBatches]);

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
          <span className="mono border border-ink px-1 text-[9px] uppercase tracking-widest text-ink">
            {`llm last $${lastLlmCost?.estimated_usd != null ? lastLlmCost.estimated_usd.toFixed(6) : "n/a"} · tok ${
              lastLlmUsage?.total_tokens != null ? lastLlmUsage.total_tokens : "n/a"
            }`}
          </span>
          <span className="mono border border-ink px-1 text-[9px] uppercase tracking-widest text-ink">
            {`llm total $${sessionLlmTotals.has_estimated_usd ? sessionLlmTotals.estimated_usd.toFixed(6) : "n/a"} · tok ${
              sessionLlmTotals.has_total_tokens ? sessionLlmTotals.total_tokens : "n/a"
            }`}
          </span>
          <span
            className={
              "mono border px-1 text-[9px] uppercase tracking-widest " +
              (serviceReady
                ? "border-signal text-signal"
                : "border-ink-red text-ink-red")
            }
            title={
              serviceReady
                ? "Backend reachable"
                : "Backend unavailable. Refresh backend service."
            }
          >
            {serviceReady ? "service ok" : "service down"}
          </span>
          <span className="eyebrow">shortcuts</span>
          <span className="kbd">G</span>
          <span className="mono text-[11px] text-ink-muted">generate</span>
          <span className="kbd">S</span>
          <span className="mono text-[11px] text-ink-muted">starters</span>
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
              onClick={() => {
                void refreshLibrary();
              }}
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
                          onDelete={handleDeleteRun}
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
                          onDelete={handleDeleteRun}
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
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <button
                    onClick={runGenerate}
                    disabled={working}
                    className="flex items-center gap-2 border border-ink bg-ink px-4 py-2 text-paper disabled:opacity-60"
                  >
                    <span aria-hidden>●</span>
                    <span className="mono text-[12px] uppercase tracking-widest">
                      {working ? "Rendering" : "Generate One Render"}
                    </span>
                    <span className="kbd !border-paper !bg-ink !text-paper">
                      G
                    </span>
                  </button>

                  <button
                    onClick={runGenerateStarters}
                    disabled={working}
                    className="flex items-center gap-2 border border-ink bg-paper px-4 py-2 text-ink disabled:opacity-60"
                  >
                    <span aria-hidden>✶</span>
                    <span className="mono text-[12px] uppercase tracking-widest">
                      Generate Starter Ideas
                    </span>
                    <span className="kbd">S</span>
                  </button>

                  <button
                    onClick={runGenerateChildren}
                    disabled={working || !runId}
                    className="flex items-center gap-2 border border-ink bg-paper px-4 py-2 text-ink disabled:opacity-40"
                  >
                    <span aria-hidden>◇</span>
                    <span className="mono text-[12px] uppercase tracking-widest">
                      {childGenMode === "llm"
                        ? "Generate LLM Children"
                        : "Generate Classic Children"}
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
                  <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">outputs</span>
                    <input
                      type="number"
                      inputMode="numeric"
                      min={1}
                      max={12}
                      value={starterCount}
                      onChange={(e) => setStarterCount(e.target.value)}
                      className="mono w-12 bg-transparent text-[12px] outline-none"
                    />
                  </label>
                  <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">max tok</span>
                    <input
                      type="number"
                      inputMode="numeric"
                      min={100}
                      max={500}
                      step={25}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(e.target.value)}
                      className="mono w-16 bg-transparent text-[12px] outline-none"
                    />
                  </label>
                  <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">note s</span>
                    <input
                      type="number"
                      inputMode="decimal"
                      min={0.1}
                      max={10}
                      step={0.1}
                      value={noteDuration}
                      onChange={(e) => setNoteDuration(e.target.value)}
                      className="mono w-14 bg-transparent text-[12px] outline-none"
                    />
                  </label>
                  <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">mode</span>
                    <select
                      value={noteMode}
                      onChange={(e) => setNoteMode(e.target.value as NoteMode)}
                      className="mono bg-transparent text-[12px] outline-none"
                    >
                      <option value="single">single</option>
                      <option value="arpeggio">arpeggio</option>
                      <option value="chord">chord</option>
                    </select>
                  </label>
                  <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">range</span>
                    <select
                      value={noteRangeMode}
                      onChange={(e) =>
                        setNoteRangeMode(e.target.value as NoteRangeMode)
                      }
                      className="mono bg-transparent text-[12px] outline-none"
                    >
                      <option value="auto">auto 2oct</option>
                      <option value="manual">manual</option>
                    </select>
                  </label>
                  {noteRangeMode === "manual" && (
                    <>
                      <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                        <span className="eyebrow">low midi</span>
                        <input
                          type="number"
                          inputMode="numeric"
                          min={24}
                          max={108}
                          value={noteLowMidi}
                          onChange={(e) => setNoteLowMidi(e.target.value)}
                          className="mono w-14 bg-transparent text-[12px] outline-none"
                        />
                      </label>
                      <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                        <span className="eyebrow">high midi</span>
                        <input
                          type="number"
                          inputMode="numeric"
                          min={24}
                          max={108}
                          value={noteHighMidi}
                          onChange={(e) => setNoteHighMidi(e.target.value)}
                          className="mono w-14 bg-transparent text-[12px] outline-none"
                        />
                      </label>
                    </>
                  )}
                  <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">child mode</span>
                    <select
                      value={childGenMode}
                      onChange={(e) =>
                        setChildGenMode(e.target.value as "classic" | "llm")
                      }
                      className="mono bg-transparent text-[12px] outline-none"
                    >
                      <option value="classic">classic</option>
                      <option value="llm">llm variants</option>
                    </select>
                  </label>
                  {childGenMode === "llm" && (
                    <>
                      <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                        <span className="eyebrow">provider</span>
                        <select
                          value={childLlmProvider}
                          onChange={(e) =>
                            setChildLlmProvider(e.target.value as LlmProvider)
                          }
                          className="mono bg-transparent text-[12px] outline-none"
                        >
                          {LLM_CHILD_PROVIDERS.map((p) => (
                            <option key={p.id} value={p.id}>
                              {p.id}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                        <span className="eyebrow">model</span>
                        <input
                          type="text"
                          value={childLlmModel}
                          onChange={(e) => setChildLlmModel(e.target.value)}
                          className="mono w-44 bg-transparent text-[12px] outline-none"
                        />
                      </label>
                      {childLlmProvider === "qwen" && (
                        <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                          <span className="eyebrow">think</span>
                          <input
                            type="checkbox"
                            checked={childLlmThink}
                            onChange={(e) => setChildLlmThink(e.target.checked)}
                            className="h-3 w-3 accent-ink"
                          />
                        </label>
                      )}
                      <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                        <span className="eyebrow">llm var</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={childLlmVariationTemp}
                          onChange={(e) => setChildLlmVariationTemp(e.target.value)}
                          className="w-24 accent-ink"
                          title="Variation amount from mother sound (0-1)"
                        />
                        <span className="mono tabular text-[11px] text-ink-muted">
                          {Number(childLlmVariationTemp).toFixed(2)}
                        </span>
                      </label>
                    </>
                  )}
                  <label className="flex items-center gap-2 border border-rule bg-paper px-3 py-2">
                    <span className="eyebrow">reverb</span>
                    <input
                      type="number"
                      inputMode="decimal"
                      min={0}
                      max={1}
                      step={0.05}
                      value={reverbMix}
                      onChange={(e) => setReverbMix(e.target.value)}
                      className="mono w-12 bg-transparent text-[12px] outline-none"
                      title="Global wet mix (0-1)"
                    />
                  </label>
                </div>

                {childBatches.length > 0 && (
                  <div className="mt-3 border border-rule bg-paper">
                    <div className="flex items-center justify-between border-b border-rule px-3 py-1.5">
                      <span className="eyebrow">
                        generated batches
                        {generatedChildrenMotherId
                          ? ` · mother #${generatedChildrenMotherId.slice(-8)}`
                          : ""}
                      </span>
                      <span className="mono tabular text-[10px] text-ink-muted">
                        {activeBatch?.children.length ?? 0} · {activeBatch?.mode ?? noteMode} ·{" "}
                        {noteDuration}s
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1 border-b border-rule px-2 py-1.5">
                      {childBatches.map((batch, index) => (
                        <div key={batch.id} className="flex items-center">
                          <button
                            type="button"
                            onClick={() => setActiveBatchId(batch.id)}
                            className={
                              "mono border px-2 py-1 text-[10px] uppercase tracking-widest " +
                              (activeBatch?.id === batch.id
                                ? "border-ink-red bg-paper-3 text-ink"
                                : "border-rule bg-paper text-ink-muted hover:text-ink")
                            }
                            title={
                              batch.motherRunId
                                ? `mother #${batch.motherRunId.slice(-8)}`
                                : "starter batch"
                            }
                          >
                            {batch.kind === "starter"
                              ? `Starter #${index + 1}`
                              : batch.kind === "child-classic"
                              ? `Classic Child #${index + 1}`
                              : `LLM Child #${index + 1}`}
                            {batch.motherRunId ? ` · ${batch.motherRunId.slice(-8)}` : ""}
                          </button>
                          <button
                            type="button"
                            onClick={() => removeChildBatch(batch.id)}
                            className="mono border-y border-r border-rule px-1 py-1 text-[10px] text-ink-muted hover:text-ink-red"
                            title="Close batch"
                            aria-label="Close batch"
                          >
                            x
                          </button>
                        </div>
                      ))}
                      <button
                        type="button"
                        onClick={clearChildBatches}
                        className="mono ml-auto border border-rule px-2 py-1 text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink-red"
                      >
                        clear all
                      </button>
                    </div>
                    <div className="border-b border-rule px-3 py-1.5">
                      <span className="mono text-[10px] text-ink-muted">
                        mother{" "}
                        {generatedChildrenMotherId
                          ? `#${generatedChildrenMotherId.slice(-8)}`
                          : runId
                          ? `#${runId.slice(-8)}`
                          : "—"}
                        {activeBatch?.variation != null
                          ? ` · llm var ${activeBatch.variation.toFixed(2)}`
                          : ""}
                      </span>
                    </div>
                    <div className="grid grid-cols-1 gap-px bg-rule sm:grid-cols-2 xl:grid-cols-3">
                      {(activeBatch?.children ?? []).map((child) => (
                        <button
                          key={child.run_id}
                          type="button"
                          onClick={() => loadStarterChild(child)}
                          className={
                            "bg-paper px-3 py-2 text-left hover:bg-paper-3 " +
                            (child.run_id === runId ? "outline outline-1 outline-ink-red" : "")
                          }
                        >
                          <div className="flex items-center justify-between">
                            <span className="mono text-[11px] uppercase tracking-widest">
                              {child.derived_from ? "child" : "starter"}
                            </span>
                            <span className="mono tabular text-[10px] text-ink-muted">
                              #{child.child_index}
                            </span>
                          </div>
                          <div className="mono mt-1 text-[10px] text-ink-muted">
                            {modeLabel(child.note_mode ?? child.starter_type)} · {child.duration_sec}s · {child.run_id.slice(-8)}
                          </div>
                          {child.derived_from && (
                            <div className="mono mt-1 text-[10px] text-ink-muted">
                              from #{child.derived_from.slice(-8)}
                            </div>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

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
                <Row label="sampling" value={`T=0.8 · top-p=0.9 · max ${maxTokens} tok`} />
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
                      <div className="mono mt-1 text-[11px] text-ink-muted">
                        cost{" "}
                        {meta.llm_cost?.estimated_usd != null
                          ? `$${meta.llm_cost.estimated_usd.toFixed(6)}`
                          : "n/a"}
                        {" · "}
                        tok{" "}
                        {meta.llm_usage?.total_tokens != null
                          ? meta.llm_usage.total_tokens
                          : "n/a"}
                      </div>
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
                if (meta.kind === "starter" || meta.kind === "child") {
                  return (
                    <div className="border-t border-rule-2 bg-paper-3/40 px-3 py-2">
                      <div className="eyebrow">{meta.kind === "child" ? "child run" : "starter run"}</div>
                      <div className="mono tabular mt-1 text-[11px]">
                        {modeLabel(meta.note_mode ?? meta.starter_type)} · child {meta.child_index} ·{" "}
                        {meta.duration_sec ?? 1.5}s
                      </div>
                      {meta.kind === "child" && meta.derived_from && (
                        <div className="mono mt-1 text-[11px] text-ink-muted">
                          mother #{meta.derived_from.slice(-8)}
                        </div>
                      )}
                      {meta.kind === "starter" && meta.parent_seed != null && (
                        <div className="mono mt-1 text-[11px] text-ink-muted">
                          parent seed {meta.parent_seed}
                        </div>
                      )}
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
  onDelete,
  onToggleFavorite,
}: {
  run: RunEntryWithMeta;
  index: number;
  active: boolean;
  onLoad: (runId: string) => void;
  onDelete: (runId: string) => void;
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
          {run.meta?.kind === "starter" && (
            <span
              className="mono border border-ink-red px-1 text-[9px] uppercase tracking-widest text-ink-red"
              title={`starter · ${modeLabel(run.meta.note_mode ?? run.meta.starter_type)} · ${run.meta.duration_sec ?? 1.5}s`}
            >
              starter
            </span>
          )}
          {run.meta?.kind === "child" && (
            <span
              className="mono border border-ink-red px-1 text-[9px] uppercase tracking-widest text-ink-red"
              title={`child · mother #${run.meta.derived_from?.slice(-8) ?? "unknown"} · ${modeLabel(run.meta.note_mode ?? run.meta.starter_type)}`}
            >
              child
            </span>
          )}
        </div>
        <div className="mono text-[10px] text-ink-muted">
          #{meta.hash} ·{" "}
          {run.meta?.kind === "edit"
            ? `${run.meta.provider} · ${run.meta.model}`
            : run.meta?.kind === "manual_edit"
            ? "hand-edited · csound"
            : run.meta?.kind === "starter"
            ? `starter · ${modeLabel(run.meta.note_mode ?? run.meta.starter_type)} · ${run.meta.duration_sec ?? 1.5}s`
            : run.meta?.kind === "child"
            ? `child · ${modeLabel(run.meta.note_mode ?? run.meta.starter_type)} · ${run.meta.duration_sec ?? 1.5}s`
            : ".csd + .wav"}
        </div>
      </div>
      <button
        type="button"
        aria-label="Delete render"
        title="Delete render"
        onClick={(e) => {
          e.stopPropagation();
          onDelete(run.run_id);
        }}
        className="mono flex h-5 w-5 items-center justify-center text-[13px] leading-none text-ink-muted opacity-0 transition-colors hover:text-ink-red group-hover:opacity-100"
      >
        ×
      </button>
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
