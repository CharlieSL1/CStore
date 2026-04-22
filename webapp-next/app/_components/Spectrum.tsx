"use client";

import { useEffect, useRef } from "react";

type Props = {
  analyser: AnalyserNode | null;
  playing: boolean;
  /** Shown when there's no audio yet. */
  placeholder?: string;
};

/**
 * An oscilloscope-meets-bar-spectrum rendered on a single canvas. The style is
 * deliberately "instrument panel": ink strokes on paper, hairline baseline,
 * ticks every 10%, no glow, no gradient.
 */
export default function Spectrum({ analyser, playing, placeholder }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  // High-DPI sizing that responds to element size changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const paper = "#ece4d3";
    const ink = "#18161a";
    const rule = "#b9ae96";
    const red = "#b1321c";

    const drawIdle = () => {
      const { width: w, height: h } = canvas;
      ctx.fillStyle = paper;
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = rule;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      ctx.setLineDash([]);

      if (placeholder) {
        ctx.fillStyle = ink;
        ctx.font = `${(12 * (window.devicePixelRatio || 1)).toFixed(0)}px "JetBrains Mono", monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(placeholder, w / 2, h / 2 - 14 * (window.devicePixelRatio || 1));
      }
    };

    if (!analyser || !playing) {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      drawIdle();
      return;
    }

    const bins = analyser.frequencyBinCount;
    const freqData = new Uint8Array(bins);
    const timeData = new Uint8Array(analyser.fftSize);

    const draw = () => {
      const { width: w, height: h } = canvas;
      analyser.getByteFrequencyData(freqData);
      analyser.getByteTimeDomainData(timeData);

      // Paper wash with a touch of persistence so bars "decay" naturally.
      ctx.fillStyle = "rgba(236, 228, 211, 0.55)";
      ctx.fillRect(0, 0, w, h);

      // Bars (bottom half-ish): ink strokes — no neon.
      const barCount = Math.min(bins, 64);
      const barWidth = w / barCount;
      ctx.fillStyle = ink;
      for (let i = 0; i < barCount; i++) {
        const v = freqData[Math.floor((i / barCount) * bins)] / 255;
        const bh = v * h * 0.88;
        ctx.fillRect(i * barWidth + 1, h - bh, barWidth - 2, bh);
      }

      // Oscilloscope overlay in ink-red.
      ctx.strokeStyle = red;
      ctx.lineWidth = Math.max(1, 1.2 * (window.devicePixelRatio || 1));
      ctx.beginPath();
      const slice = w / timeData.length;
      for (let i = 0; i < timeData.length; i++) {
        const y = (timeData[i] / 255) * h;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * slice, y);
      }
      ctx.stroke();

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [analyser, playing, placeholder]);

  return (
    <canvas
      ref={canvasRef}
      className="block h-[240px] w-full border border-rule bg-paper-2"
    />
  );
}
