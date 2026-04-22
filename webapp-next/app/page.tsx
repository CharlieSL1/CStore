import Console from "./_components/Console";

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
