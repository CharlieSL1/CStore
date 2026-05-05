import Console from "./_components/Console";

const faqItems = [
  {
    question: "Why CStore?",
    answer:
      "CStore exists because most text-to-music systems output final audio but not an editable creative artifact. Instead of generating only waveforms, CStore generates editable Csound .csd specifications that are inspectable, revision-friendly, versionable, and reproducible.",
  },
  {
    question:
      "If large AI models can generate better sound, why do we need CStore?",
    answer:
      "CStore is designed for a different goal: controllable workflow, not one-shot polish. Large models are strong at perceptual media generation, while CStore focuses on generating a transparent specification you can inspect, edit at parameter level, and deterministically re-render.",
  },
  {
    question:
      "Is CStore a replacement for GPT, Claude, Suno, MusicGen, or Sora-style systems?",
    answer:
      "No. CStore is complementary. Frontier models are useful for high-quality media output and broad reasoning. CStore is for local, editable, executable, and reproducible music specification generation.",
  },
  {
    question: "What does “editable” mean in CStore?",
    answer:
      "The output is a Csound program, not just a sound file. You can directly modify synthesis parameters, envelopes, filters, modulation, event timing, score values, and instrument structure in the generated .csd.",
  },
  {
    question: "How do you evaluate CStore?",
    answer:
      "CStore uses layered usability metrics: structural validity, render success, and audible output. The baseline reports are 61% structural validity, 32% render success, and 19% audible output in a representative 100-sample batch (seed 42), with major improvements after expert-curated fine-tuning.",
  },
];

/**
 * The page is a server component that ships only markup + tokens. All
 * interactivity (generate, playback, FFT, copy, library) lives in one client
 * island — <Console /> — to keep the initial paint static and typographic.
 */
export default function Page() {
  return (
    <main className="relative z-10 mx-auto max-w-[1320px] px-6 pb-24 pt-8 md:px-10">
      {/* ———— Masthead ————————————————————————————————————————————— */}
      <header className="flex items-end justify-between border-b border-ink pb-5">
        <div className="flex items-end gap-5">
          <div className="mono text-[11px] leading-tight text-ink-muted">
            <div>VOL. I</div>
            <div>NO. 012</div>
          </div>
          <div className="h-12 w-px bg-ink/40" aria-hidden />
          <div>
            <div className="eyebrow">A Console for Interpretable Music Models</div>
            <h1 className="display mt-1 text-[64px] md:text-[84px]">
              CStore<span className="text-ink-red">.</span>
            </h1>
          </div>
        </div>
        <div className="hidden flex-col items-end gap-1 md:flex">
          <div className="eyebrow">Default checkpoint</div>
          <div className="mono text-sm">Cstore_V1.0.2 / best</div>
          <div className="mono text-[11px] text-ink-muted">
            GPT-2 · 8L · 384h · 33M params · 512 ctx
          </div>
        </div>
      </header>

      {/* ———— Strapline row ———————————————————————————————————————— */}
      <section className="grid grid-cols-12 gap-6 border-b border-rule-2 py-5 text-sm">
        <p className="col-span-12 text-[15px] leading-relaxed text-ink-2 md:col-span-8">
          CStore reframes text-to-music away from one-shot waveforms and toward a
          <em className="mx-1 italic">human-readable Csound specification</em>:
          auditable, diff-able, and editable before it ever becomes sound. This
          console is the local control room — press{" "}
          <span className="kbd">G</span> to draft a new{" "}
          <span className="mono">.csd</span>, audition the render, and send the
          source to the Csound IDE when you want to keep working by hand.
        </p>
        <div className="col-span-6 md:col-span-2">
          <div className="eyebrow">Struct / Render / Sound</div>
          <div className="mono tabular mt-1 text-sm">73% · 56% · 54%</div>
        </div>
        <div className="col-span-6 md:col-span-2">
          <div className="eyebrow">Dataset</div>
          <div className="mono tabular mt-1 text-sm">27,104 .csd</div>
        </div>
      </section>

      {/* ———— Console (client island) ——————————————————————————————— */}
      <Console />

      {/* ———— FAQ ——————————————————————————————————————————————————— */}
      <section className="border-b border-rule-2 py-5">
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 md:col-span-4">
            <div className="eyebrow">Frequently Asked Questions</div>
            <p className="mt-1 text-[15px] leading-relaxed text-ink-2">
              Core positioning for CStore as an editable and reproducible
              text-to-music workflow.
            </p>
          </div>
          <div className="col-span-12 space-y-3 md:col-span-8">
            {faqItems.map((item) => (
              <details
                key={item.question}
                className="border border-ink/25 bg-paper-soft px-4 py-3"
              >
                <summary className="cursor-pointer list-none pr-6 text-[15px] leading-relaxed text-ink">
                  <span className="mono mr-2 text-[11px] uppercase tracking-[0.16em] text-ink-muted">
                    Q
                  </span>
                  {item.question}
                </summary>
                <p className="mt-3 border-t border-ink/20 pt-3 text-sm leading-relaxed text-ink-2">
                  <span className="mono mr-2 text-[11px] uppercase tracking-[0.16em] text-ink-muted">
                    A
                  </span>
                  {item.answer}
                </p>
              </details>
            ))}
          </div>
        </div>
      </section>

      {/* ———— Colophon ———————————————————————————————————————————— */}
      <footer className="mt-10 grid grid-cols-12 gap-6 border-t border-ink pt-5 text-sm">
        <div className="col-span-12 md:col-span-4">
          <div className="eyebrow">Colophon</div>
          <p className="mt-1 leading-relaxed text-ink-2">
            Set in <span className="italic">Instrument Serif</span>, Inter, and
            JetBrains Mono. Rendered by Next.js 16 · React 19 · Tailwind 4.
            Audio rendering requires local{" "}
            <a
              href="https://csound.com/"
              target="_blank"
              rel="noreferrer"
              className="underline decoration-ink-red decoration-2 underline-offset-4"
            >
              Csound 6.18
            </a>
            .
          </p>
        </div>
        <div className="col-span-6 md:col-span-4">
          <div className="eyebrow">Authors</div>
          <p className="mono mt-1 leading-relaxed">
            Li (Charlie) Shi — cshi@berklee.edu
            <br />
            Richard Boulanger — rboulanger@berklee.edu
          </p>
        </div>
        <div className="col-span-6 md:col-span-4 md:text-right">
          <div className="eyebrow">Repository</div>
          <p className="mono mt-1">
            <a
              href="https://github.com/CharlieSL1/CStore"
              target="_blank"
              rel="noreferrer"
              className="underline decoration-ink-red decoration-2 underline-offset-4"
            >
              github.com/CharlieSL1/CStore
            </a>
          </p>
          <p className="mono mt-1 text-[11px] text-ink-muted">
            MIT License · IEEE CAI 2026
          </p>
        </div>
      </footer>
    </main>
  );
}
