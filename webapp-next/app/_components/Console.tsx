"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  batchTag?: string;
  expectedCount?: number;
  createdAt: number;
  children: StarterChild[];
};

type UserFolder = {
  id: string;
  name: string;
  runIds: string[];
  collapsed: boolean;
};

type PendingBatch = {
  id: string;
  kind: ChildBatchKind;
  motherRunId: string | null;
  expectedCount: number;
  mode: string;
  variation: number | null;
  batchTag: string;
  createdAt: number;
  startedAtMs: number;
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
  gemini: "gemini-3.1-pro-preview",
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
  const [pendingBatches, setPendingBatches] = useState<PendingBatch[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [selectionAnchorRunId, setSelectionAnchorRunId] = useState<string | null>(null);
  const [userFolders, setUserFolders] = useState<UserFolder[]>([]);
  const [collapsedMotherLibrary, setCollapsedMotherLibrary] = useState<Record<string, boolean>>({});
  const [collapsedBatchGroups, setCollapsedBatchGroups] = useState<Record<string, boolean>>({});
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

  const createBatchTag = useCallback((prefix: string) => {
    return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
  }, []);

  const removeRunIdFromState = useCallback((id: string) => {
    setSelectedRunIds((prev) => prev.filter((runId) => runId !== id));
    setSelectionAnchorRunId((prev) => (prev === id ? null : prev));
    setUserFolders((prev) =>
      prev
        .map((folder) => ({
          ...folder,
          runIds: folder.runIds.filter((runId) => runId !== id),
        }))
        .filter((folder) => folder.runIds.length > 0)
    );
  }, []);

  const appendChildBatch = useCallback((batch: Omit<ChildBatch, "id" | "createdAt">) => {
    const id = `batch_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
    const createdAt = Date.now();
    setChildBatches((prev) => {
      if (batch.batchTag) {
        const existingIndex = prev.findIndex((b) => b.batchTag === batch.batchTag);
        if (existingIndex !== -1) {
          const next = [...prev];
          next[existingIndex] = { ...next[existingIndex], ...batch };
          return next;
        }
      }
      return [...prev, { ...batch, id, createdAt }];
    });
    setActiveBatchId((prev) => prev ?? id);
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
    setPendingBatches([]);
  }, []);

  const addPendingBatch = useCallback(
    (batch: Omit<PendingBatch, "id" | "createdAt" | "startedAtMs">) => {
      const id = `pending_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
      const createdAt = Date.now();
      setPendingBatches((prev) => [
        ...prev,
        { ...batch, id, createdAt, startedAtMs: Date.now() },
      ]);
      return id;
    },
    []
  );

  const removePendingBatch = useCallback((id: string) => {
    setPendingBatches((prev) => prev.filter((batch) => batch.id !== id));
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
    const batchTag = createBatchTag("starter");
    const pendingId = addPendingBatch({
      kind: "starter",
      motherRunId: null,
      expectedCount: parsedCount,
      mode: modeLabel(noteMode),
      variation: null,
      batchTag,
    });
    try {
      const data = await generateStarters({
        count: parsedCount,
        seed: parsedSeed,
        checkpoint: activeModelLabel || undefined,
        batch_tag: batchTag,
        max_tokens: parsedMaxTokens,
        note_duration: parsedNoteDuration,
        note_mode: noteMode,
        note_range_mode: noteRangeMode,
        note_low_midi: Math.min(parsedLowMidi, parsedHighMidi),
        note_high_midi: Math.max(parsedLowMidi, parsedHighMidi),
      });
      removePendingBatch(pendingId);
      appendChildBatch({
        kind: "starter",
        motherRunId: null,
        mode: modeLabel(data.note_mode ?? noteMode),
        variation: null,
        batchTag: data.batch_tag ?? batchTag,
        expectedCount: parsedCount,
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
      removePendingBatch(pendingId);
    }
  }, [
    activeModelLabel,
    addPendingBatch,
    appendChildBatch,
    createBatchTag,
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
    removePendingBatch,
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
    const batchTag = createBatchTag(childGenMode === "llm" ? "child_llm" : "child_classic");
    const pendingId = addPendingBatch({
      kind: childGenMode === "llm" ? "child-llm" : "child-classic",
      motherRunId: runId,
      expectedCount: parsedCount,
      mode: modeLabel(noteMode),
      variation: childGenMode === "llm" ? parsedVariationTemp : null,
      batchTag,
    });
    try {
      const data =
        childGenMode === "llm"
          ? await generateChildrenLlm({
              source_run_id: runId,
              count: parsedCount,
              batch_tag: batchTag,
              provider: childLlmProvider,
              model: childLlmModel.trim(),
              think: childLlmProvider === "qwen" ? childLlmThink : undefined,
              variation_temperature: parsedVariationTemp,
            })
          : await generateChildren({
              source_run_id: runId,
              count: parsedCount,
              batch_tag: batchTag,
              note_duration: parsedNoteDuration,
              note_mode: noteMode,
              note_range_mode: noteRangeMode,
              note_low_midi: Math.min(parsedLowMidi, parsedHighMidi),
              note_high_midi: Math.max(parsedLowMidi, parsedHighMidi),
            });
      removePendingBatch(pendingId);
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
        batchTag: data.batch_tag ?? batchTag,
        expectedCount: parsedCount,
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
      removePendingBatch(pendingId);
    }
  }, [
    addPendingBatch,
    childGenMode,
    appendChildBatch,
    createBatchTag,
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
    removePendingBatch,
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
    if (!child.csd) {
      void loadRun(child.run_id);
      return;
    }
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
  }, [loadRun]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || !audio.src) return;
    ensureAudioGraph();
    audioCtxRef.current?.resume();
    if (audio.paused) {
      audio.currentTime = 0;
      audio.play();
    }
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
        removeRunIdFromState(id);
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
    [refreshLibrary, removeRunIdFromState, runId]
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
  const batchGroups = useMemo(() => {
    const groups = new Map<string, { id: string; label: string; batches: ChildBatch[] }>();
    for (const batch of childBatches) {
      const key = batch.motherRunId ? `mother:${batch.motherRunId}` : "starter";
      const label = batch.motherRunId ? `mother #${batch.motherRunId.slice(-8)}` : "starter batches";
      const current = groups.get(key);
      if (current) {
        current.batches.push(batch);
      } else {
        groups.set(key, { id: key, label, batches: [batch] });
      }
    }
    return Array.from(groups.values());
  }, [childBatches]);

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
  const visibleFolderRuns = useMemo(
    () =>
      userFolders.map((folder) => ({
        ...folder,
        runs: folder.runIds
          .map((id) => runs.find((r) => r.run_id === id))
          .filter(Boolean) as RunEntryWithMeta[],
      })),
    [runs, userFolders]
  );
  const folderRunIds = useMemo(() => {
    const ids = new Set<string>();
    for (const folder of visibleFolderRuns) {
      for (const run of folder.runs) ids.add(run.run_id);
    }
    return ids;
  }, [visibleFolderRuns]);
  const libraryRecentUnfoldered = recentRuns.filter((r) => !folderRunIds.has(r.run_id));
  const recentChildRuns = libraryRecentUnfoldered.filter(
    (r) => r.meta?.kind === "child" && r.meta?.derived_from
  );
  const recentNonChildRuns = libraryRecentUnfoldered.filter(
    (r) => !(r.meta?.kind === "child" && r.meta?.derived_from)
  );
  const recentChildrenByMother = useMemo(() => {
    const map = new Map<string, RunEntryWithMeta[]>();
    for (const run of recentChildRuns) {
      const mother = run.meta?.derived_from;
      if (!mother) continue;
      const list = map.get(mother) ?? [];
      list.push(run);
      map.set(mother, list);
    }
    return Array.from(map.entries()).sort((a, b) => b[0].localeCompare(a[0]));
  }, [recentChildRuns]);
  const librarySelectableOrder = useMemo(() => {
    const out: string[] = [];
    for (const run of favoriteRuns) out.push(run.run_id);
    for (const folder of visibleFolderRuns) {
      for (const run of folder.runs) out.push(run.run_id);
    }
    for (const [_, grouped] of recentChildrenByMother) {
      for (const run of grouped) out.push(run.run_id);
    }
    for (const run of recentNonChildRuns) out.push(run.run_id);
    return out;
  }, [favoriteRuns, visibleFolderRuns, recentChildrenByMother, recentNonChildRuns]);
  const libraryIndexMap = useMemo(
    () =>
      new Map(
        librarySelectableOrder.map((id, idx) => [id, librarySelectableOrder.length - idx])
      ),
    [librarySelectableOrder]
  );

  const handleLibraryRowInteract = useCallback(
    (runId: string, modifiers: { shift: boolean; meta: boolean }) => {
      if (!modifiers.shift && !modifiers.meta) {
        setSelectionAnchorRunId(runId);
        setSelectedRunIds([]);
        void loadRun(runId);
        return;
      }
      if (modifiers.shift && selectionAnchorRunId) {
        const anchorIndex = librarySelectableOrder.indexOf(selectionAnchorRunId);
        const clickIndex = librarySelectableOrder.indexOf(runId);
        if (anchorIndex !== -1 && clickIndex !== -1) {
          const [start, end] =
            anchorIndex < clickIndex ? [anchorIndex, clickIndex] : [clickIndex, anchorIndex];
          const range = librarySelectableOrder.slice(start, end + 1);
          setSelectedRunIds((prev) =>
            Array.from(new Set([...(modifiers.meta ? prev : []), ...range]))
          );
          return;
        }
      }
      setSelectionAnchorRunId(runId);
      setSelectedRunIds((prev) =>
        prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]
      );
    },
    [librarySelectableOrder, loadRun, selectionAnchorRunId]
  );

  const handleCreateFolderFromSelected = useCallback(() => {
    if (!selectedRunIds.length) return;
    const name = window.prompt("Folder name for selected runs?");
    const trimmed = name?.trim();
    if (!trimmed) return;
    setUserFolders((prev) => {
      const id = `folder_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
      const selected = new Set(selectedRunIds);
      const nextPrev = prev.map((folder) => ({
        ...folder,
        runIds: folder.runIds.filter((runId) => !selected.has(runId)),
      }));
      return [...nextPrev.filter((f) => f.runIds.length > 0), { id, name: trimmed, runIds: [...selectedRunIds], collapsed: false }];
    });
    setSelectedRunIds([]);
  }, [selectedRunIds]);

  const toggleFolderCollapse = useCallback((folderId: string) => {
    setUserFolders((prev) =>
      prev.map((folder) =>
        folder.id === folderId ? { ...folder, collapsed: !folder.collapsed } : folder
      )
    );
  }, []);

  const handleDeleteSelectedRuns = useCallback(async () => {
    if (!selectedRunIds.length) return;
    if (!window.confirm(`Delete ${selectedRunIds.length} selected run(s)?`)) return;
    let deleted = 0;
    let failed = 0;
    for (const id of selectedRunIds) {
      try {
        await deleteRun(id);
        deleted += 1;
        removeRunIdFromState(id);
        setRuns((prev) => prev.filter((r) => r.run_id !== id));
      } catch {
        failed += 1;
      }
    }
    setChildBatches((prev) =>
      prev
        .map((batch) => ({
          ...batch,
          children: batch.children.filter((child) => !selectedRunIds.includes(child.run_id)),
        }))
        .filter((batch) => batch.children.length > 0)
    );
    setSelectedRunIds([]);
    await refreshLibrary(true);
    if (failed > 0) {
      setStatus({
        kind: "err",
        message: `Deleted ${deleted} selected run(s), ${failed} failed`,
      });
      return;
    }
    setStatus({ kind: "ok", message: `Deleted ${deleted} selected run(s)` });
  }, [refreshLibrary, removeRunIdFromState, selectedRunIds]);

  useEffect(() => {
    setSelectedRunIds((prev) => prev.filter((runId) => runs.some((run) => run.run_id === runId)));
  }, [runs]);

  // ——— Keyboard shortcuts — only when focus isn't in a text input —————
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const withMeta = e.metaKey || e.ctrlKey;
      if (withMeta && e.key === "Backspace" && selectedRunIds.length > 0) {
        e.preventDefault();
        void handleDeleteSelectedRuns();
        return;
      }
      if (
        withMeta &&
        e.shiftKey &&
        (e.key === "f" || e.key === "F") &&
        selectedRunIds.length > 0
      ) {
        e.preventDefault();
        handleCreateFolderFromSelected();
        return;
      }
      if (withMeta || e.altKey) return;
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
  }, [
    copyCsd,
    csd,
    handleCreateFolderFromSelected,
    handleDeleteSelectedRuns,
    runGenerate,
    runGenerateStarters,
    runId,
    selectedRunIds.length,
    togglePlay,
  ]);

  useEffect(() => {
    if (!pendingBatches.length) return;
    const timer = window.setInterval(() => {
      void refreshLibrary(true);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [pendingBatches, refreshLibrary]);

  useEffect(() => {
    if (!pendingBatches.length) return;
    setChildBatches((prev) => {
      let next = [...prev];
      for (const pending of pendingBatches) {
        const matchedRuns = runs.filter((run) => {
          const meta = run.meta;
          if (!meta || meta.batch_tag !== pending.batchTag) return false;
          if (pending.kind === "starter") return meta.kind === "starter";
          if (pending.motherRunId) return meta.kind === "child" && meta.derived_from === pending.motherRunId;
          return meta.kind === "child";
        });
        const children: StarterChild[] = matchedRuns.map((run, index) => ({
          csd: "",
          run_id: run.run_id,
          csd_url: `/generated/${encodeURIComponent(run.run_id)}/output.csd`,
          wav_url: `/generated/${encodeURIComponent(run.run_id)}/output.wav`,
          starter_type: modeLabel(run.meta?.starter_type) as NoteMode,
          note_mode: modeLabel(run.meta?.note_mode ?? run.meta?.starter_type) as NoteMode,
          child_index: run.meta?.child_index ?? index + 1,
          duration_sec: (run.meta?.duration_sec ?? Number(noteDuration)) || 1.5,
          derived_from: run.meta?.derived_from,
        }));
        const batchPatch: Omit<ChildBatch, "id" | "createdAt"> = {
          kind: pending.kind,
          motherRunId: pending.motherRunId,
          mode: pending.mode,
          variation: pending.variation,
          batchTag: pending.batchTag,
          expectedCount: pending.expectedCount,
          children,
        };
        const idx = next.findIndex((batch) => batch.batchTag === pending.batchTag);
        if (idx !== -1) {
          const existing = next[idx];
          const mergedChildren =
            batchPatch.children.length > existing.children.length
              ? batchPatch.children
              : existing.children;
          next[idx] = { ...existing, ...batchPatch, children: mergedChildren };
        } else {
          next.push({
            id: `live_${pending.id}`,
            createdAt: pending.createdAt,
            ...batchPatch,
          });
        }
      }
      return next;
    });
  }, [noteDuration, pendingBatches, runs]);

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
            <div className="flex items-center gap-2">
              <span className="eyebrow-ink">Library</span>
              {selectedRunIds.length > 0 && (
                <span className="mono border border-rule px-1 text-[9px] uppercase tracking-widest text-ink-muted">
                  {selectedRunIds.length} selected
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCreateFolderFromSelected}
                disabled={!selectedRunIds.length}
                className="mono border border-rule bg-paper px-2 py-1 text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink disabled:opacity-40"
              >
                folder
              </button>
              <button
                onClick={handleDeleteSelectedRuns}
                disabled={!selectedRunIds.length}
                className="mono border border-rule bg-paper px-2 py-1 text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink-red disabled:opacity-40"
              >
                delete
              </button>
              <button
                onClick={() => {
                  void refreshLibrary();
                }}
                className="mono text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink-red"
              >
                refresh
              </button>
            </div>
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
                      {favoriteRuns.map((r) => (
                        <LibraryRow
                          key={r.run_id}
                          run={r}
                          index={libraryIndexMap.get(r.run_id) ?? 0}
                          active={r.run_id === runId}
                          selected={selectedRunIds.includes(r.run_id)}
                          onInteract={handleLibraryRowInteract}
                          onDelete={handleDeleteRun}
                          onToggleFavorite={toggleFavorite}
                        />
                      ))}
                    </ol>
                  </div>
                )}

                {visibleFolderRuns.map((folder) => (
                  <div key={folder.id}>
                    <button
                      type="button"
                      onClick={() => toggleFolderCollapse(folder.id)}
                      className="flex w-full items-center justify-between border-b border-rule-2 bg-paper-3/50 px-3 py-1.5 text-left"
                    >
                      <span className="eyebrow">{folder.collapsed ? "▸" : "▾"} {folder.name}</span>
                      <span className="mono tabular text-[10px] text-ink-muted">{folder.runs.length}</span>
                    </button>
                    {!folder.collapsed && (
                      <ol>
                        {folder.runs.map((r) => (
                          <LibraryRow
                            key={r.run_id}
                            run={r}
                            index={libraryIndexMap.get(r.run_id) ?? 0}
                            active={r.run_id === runId}
                            selected={selectedRunIds.includes(r.run_id)}
                            onInteract={handleLibraryRowInteract}
                            onDelete={handleDeleteRun}
                            onToggleFavorite={toggleFavorite}
                          />
                        ))}
                      </ol>
                    )}
                  </div>
                ))}

                <div>
                  {recentChildrenByMother.map(([motherId, grouped]) => (
                    <div key={motherId}>
                      <button
                        type="button"
                        onClick={() =>
                          setCollapsedMotherLibrary((prev) => ({
                            ...prev,
                            [motherId]: !prev[motherId],
                          }))
                        }
                        className="flex w-full items-center justify-between border-b border-rule-2 bg-paper-3/20 px-3 py-1.5 text-left"
                      >
                        <span className="eyebrow">
                          {collapsedMotherLibrary[motherId] ? "▸" : "▾"} mother #{motherId.slice(-8)}
                        </span>
                        <span className="mono tabular text-[10px] text-ink-muted">{grouped.length}</span>
                      </button>
                      {!collapsedMotherLibrary[motherId] && (
                        <ol>
                          {grouped.map((r) => (
                            <LibraryRow
                              key={r.run_id}
                              run={r}
                              index={libraryIndexMap.get(r.run_id) ?? 0}
                              active={r.run_id === runId}
                              selected={selectedRunIds.includes(r.run_id)}
                              onInteract={handleLibraryRowInteract}
                              onDelete={handleDeleteRun}
                              onToggleFavorite={toggleFavorite}
                            />
                          ))}
                        </ol>
                      )}
                    </div>
                  ))}

                  {favoriteRuns.length > 0 && (
                    <div className="flex items-center justify-between border-b border-rule-2 bg-paper-2 px-3 py-1.5">
                      <span className="eyebrow">recent</span>
                      <span className="mono tabular text-[10px] text-ink-muted">
                        {recentNonChildRuns.length}
                      </span>
                    </div>
                  )}
                  {recentNonChildRuns.length === 0 ? (
                    <p className="mono px-3 py-3 text-[11px] text-ink-muted">
                      Everything here is a favourite — generate a new
                      timbre with <span className="kbd">G</span>.
                    </p>
                  ) : (
                    <ol>
                      {recentNonChildRuns.map((r) => (
                        <LibraryRow
                          key={r.run_id}
                          run={r}
                          index={libraryIndexMap.get(r.run_id) ?? 0}
                          active={r.run_id === runId}
                          selected={selectedRunIds.includes(r.run_id)}
                          onInteract={handleLibraryRowInteract}
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
                    className="flex items-center gap-2 border border-ink bg-paper px-4 py-2 text-ink transition-colors active:bg-paper-3 disabled:opacity-60"
                  >
                    <span aria-hidden>●</span>
                    <span className="mono text-[12px] uppercase tracking-widest">
                      {working ? "Rendering" : "Generate One Render"}
                    </span>
                    <span className="kbd">G</span>
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

                {(childBatches.length > 0 || pendingBatches.length > 0) && (
                  <div className="mt-3 border border-rule bg-paper">
                    <div className="flex items-center justify-between border-b border-rule px-3 py-1.5">
                      <span className="eyebrow">
                        generated batches
                        {generatedChildrenMotherId
                          ? ` · mother #${generatedChildrenMotherId.slice(-8)}`
                          : ""}
                      </span>
                      <span className="mono tabular text-[10px] text-ink-muted">
                        {childBatches.length} ready · {pendingBatches.length} pending
                      </span>
                    </div>
                    <div className="space-y-2 border-b border-rule px-2 py-2">
                      {batchGroups.map((group) => (
                        <div key={group.id} className="border border-rule">
                          <button
                            type="button"
                            onClick={() =>
                              setCollapsedBatchGroups((prev) => ({
                                ...prev,
                                [group.id]: !prev[group.id],
                              }))
                            }
                            className="flex w-full items-center justify-between bg-paper-3/30 px-2 py-1.5 text-left"
                          >
                            <span className="eyebrow">
                              {collapsedBatchGroups[group.id] ? "▸" : "▾"} {group.label}
                            </span>
                            <span className="mono tabular text-[10px] text-ink-muted">
                              {group.batches.reduce((sum, batch) => sum + batch.children.length, 0)}
                            </span>
                          </button>
                          {!collapsedBatchGroups[group.id] && (
                            <div className="space-y-1 px-2 py-2">
                              {group.batches.map((batch, index) => (
                                <div key={batch.id} className="border border-rule">
                                  <div className="flex items-center justify-between border-b border-rule bg-paper px-2 py-1">
                                    <button
                                      type="button"
                                      onClick={() => setActiveBatchId(batch.id)}
                                      className={
                                        "mono text-[10px] uppercase tracking-widest " +
                                        (activeBatch?.id === batch.id ? "text-ink-red" : "text-ink-muted")
                                      }
                                    >
                                      {batch.kind === "starter"
                                        ? `Starter #${index + 1}`
                                        : batch.kind === "child-classic"
                                        ? `Classic Child #${index + 1}`
                                        : `LLM Child #${index + 1}`}
                                      {batch.expectedCount && batch.children.length < batch.expectedCount
                                        ? ` · ${batch.children.length}/${batch.expectedCount}`
                                        : ""}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => removeChildBatch(batch.id)}
                                      className="mono text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink-red"
                                      title="Close batch"
                                      aria-label="Close batch"
                                    >
                                      close
                                    </button>
                                  </div>
                                  <div className="grid grid-cols-1 gap-px bg-rule sm:grid-cols-2 xl:grid-cols-3">
                                    {batch.children.map((child) => (
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
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                      {pendingBatches.length > 0 && (
                        <div className="border border-rule bg-paper-3/20 px-3 py-2">
                          <div className="eyebrow">pending</div>
                          <div className="mono mt-1 text-[10px] text-ink-muted">
                            {pendingBatches
                              .map(
                                (batch) =>
                                  `${batch.kind === "starter" ? "starter" : "child"} ${batch.mode} · ${batch.expectedCount} outputs`
                              )
                              .join(" · ")}
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center justify-between px-3 py-1.5">
                      <span className="mono text-[10px] text-ink-muted">
                        {activeBatch?.variation != null
                          ? `active llm var ${activeBatch.variation.toFixed(2)}`
                          : "select any generated child to load"}
                      </span>
                      <button
                        type="button"
                        onClick={clearChildBatches}
                        className="mono border border-rule px-2 py-1 text-[10px] uppercase tracking-widest text-ink-muted hover:text-ink-red"
                      >
                        clear all
                      </button>
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
              <div className="scroll-ink relative grid max-h-[420px] grid-cols-[3rem_1fr] overflow-auto">
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
                {working && <div className="hatch pointer-events-none absolute inset-0" />}
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
  selected,
  onInteract,
  onDelete,
  onToggleFavorite,
}: {
  run: RunEntryWithMeta;
  index: number;
  active: boolean;
  selected: boolean;
  onInteract: (runId: string, modifiers: { shift: boolean; meta: boolean }) => void;
  onDelete: (runId: string) => void;
  onToggleFavorite: (runId: string) => void;
}) {
  const meta = formatRunId(run.run_id);
  const starred = Boolean(run.meta?.favorite);
  return (
    <li
      className={
        "group flex cursor-pointer items-baseline gap-3 border-b border-rule-2 px-3 py-2 " +
        (selected ? "bg-paper-3/80" : active ? "bg-paper-3" : "hover:bg-paper-3/60")
      }
      onClick={(e) =>
        onInteract(run.run_id, {
          shift: e.shiftKey,
          meta: e.metaKey || e.ctrlKey,
        })
      }
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
