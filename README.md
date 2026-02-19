# CStore

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

[![Contributors](https://contrib.rocks/image?repo=CharlieSL1/CStore)](https://github.com/CharlieSL1/CStore/graphs/contributors)

**CStore: Reframing Text-to-Music from "One-Shot Waveforms" to an Interpretable, Durable, and Editable Csound Specification**

Instead of generating audio directly, CStore generates **Csound project files (.csd)** — human-readable, version-controlled, and editable before rendering.

- **Interpretable** — Auditable source code
- **Durable** — Text-based, reproducible
- **Editable** — Modify by hand or programmatically

![CStore Banner](media/CStore.gif)

---

## Quick Start

```bash
git clone https://github.com/CharlieSL1/CStore.git
cd CStore
pip install -r requirements.txt
cd model
python evaluate.py --checkpoint checkpoints/Cstore_V1.0.1/best --num_samples 100 --seed 42
```

**Requirements** — Python 3.11+, PyTorch, Transformers, [Csound](https://csound.com/) 6.18.0

---

## Usage

| Task | Command |
|------|---------|
| **Evaluate** | `python evaluate.py --checkpoint checkpoints/Cstore_V1.0.1/best --num_samples 100` |
| **Generate** | `python generate.py` |
| **Train** | `python train_finetune_expert.py` → `train_finetune_continuation.py` |

See [model/README.md](model/README.md) for details.

---

## Model

| Version | Struct | Render | Sound |
|---------|--------|--------|-------|
| **V1.0.0** (baseline) | 61% | 32% | 19% |
| **V1.0.1** (expert fine-tune) | 73% | 56% | 54% |
| **V1.0.2** (continuation) | 72% | 53% | 50% |

*Metrics: structural validity → render success → audible output (RMS > 1e-4)*

**Architecture** — GPT-2 style Transformer, 8 layers, 384 hidden, 33M params, 512 context length.

---

## Audio Samples

**[▶ Play all in browser](https://charliesl1.github.io/CStore/)** — No download, plays directly on the page.

**random_seed_gen_001**

[▶ Play](https://charliesl1.github.io/CStore/#random_seed_gen_001)

<details>
<summary>View CSD</summary>

```csd
<CsoundSynthesizer>
<CsOptions></CsOptions>
<CsInstruments>
0dbfs  = 1
gisin ftgen 0, 0, 2^10, 10, 1
instr 1
ksig randomh 400, 1800, 150
aout poscil .2, 1000+ksig, gisin
outs aout, aout
endin
instr 2
ksig randomh 400, 1800, 150
khp line 1, p3, 400
ksig atonek ksig, khp
aout poscil .2, 1000+ksig, gisin
outs aout, aout
endin
</CsInstruments>
<CsScore>
i 1 0 5
i 2 5.5 5
e
</CsScore>
</CsoundSynthesizer>
```

</details>

**random_seed_gen_002**

[▶ Play](https://charliesl1.github.io/CStore/#random_seed_gen_002)

<details>
<summary>View CSD</summary>

```csd
<CsoundSynthesizer>
<CsOptions></CsOptions>
<CsInstruments>
0dbfs  = 1
instr 1
kfreq = 220
kc1 = 5
kc2 = p4
kvrate = 6
kvdpth line 0, p3, p5
asig fmb3 .4, kfreq, kc1, kc2, kvdpth, kvrate
outs asig, asig
endin
</CsInstruments>
<CsScore>
f 1 0 32768 10 1
i 1 0 2  5 0.1
i 1 3 2 .5 .5 0.01
i 1 4 .5 .5 0.01
e
</CsScore>
</CsoundSynthesizer>
```

</details>

**random_seed_gen_003**

[▶ Play](https://charliesl1.github.io/CStore/#random_seed_gen_003)

<details>
<summary>View CSD</summary>

```csd
<CsoundSynthesizer>
<CsOptions></CsOptions>
<CsInstruments>
0dbfs  = 1
instr 1
kcps = 110
kcar = 1
kmod = p4
kndx oscil 30, .25/p3, 1
kndx ceil kndx
asig foscili .5, kcps, kcar, kmod, kndx, 1
outs asig, asig
endin
</CsInstruments>
<CsScore>
f 1 0 16384 10 1
i 1 0 10 1.414
e
</CsScore>
</CsoundSynthesizer>
```

</details>

**random_seed_gen_004**

[▶ Play](https://charliesl1.github.io/CStore/#random_seed_gen_004)

<details>
<summary>View CSD</summary>

```csd
<CsoundSynthesizer>
<CsOptions></CsOptions>
<CsInstruments>
0dbfs  = 1
instr 1
kval randomh 0, 1.2, 20
if kval >0 && kval<=.5 then kval = 1
elseif kval >.5 && kval<=1 then kval = 2
elseif kval >1 then kval = 3
endif
asig poscil .7, 440*kval, 1
outs asig, asig
endin
</CsInstruments>
<CsScore>
f1 0 16384 10 1
i 1 0 5
e
</CsScore>
</CsoundSynthesizer>
```

</details>

**user_input_seed**

[▶ Play](https://charliesl1.github.io/CStore/#user_input_seed)

<details>
<summary>View CSD</summary>

Same as random_seed_gen_003.

</details>

**variation_1**

[▶ Play](https://charliesl1.github.io/CStore/#variation_1)

<details>
<summary>View CSD</summary>

```csd
<CsoundSynthesizer>
<CsOptions></CsOptions>
<CsInstruments>
seed 7919
0dbfs  = 1
instr 1
kcps = 110
kcar = 1
kmod = p4
kndx oscil 30, .25/p3, 1
kndx ceil kndx
asig foscili .5, 220+kcar, kmod, kndx, 1
outs asig, asig
endin
</CsInstruments>
<CsScore>
f 1 0 16384 10 1
i 1 0.5 12.5 2.2624
e
</CsScore>
</CsoundSynthesizer>
```

</details>

**variation_2**

[▶ Play](https://charliesl1.github.io/CStore/#variation_2)

<details>
<summary>View CSD</summary>

```csd
<CsoundSynthesizer>
<CsOptions></CsOptions>
<CsInstruments>
seed 15838
0dbfs  = 1
instr 1
kcps = 110
kcar = 1
kmod = p4
kndx oscil 30, .25/p3, 1
kndx ceil kndx
asig foscili .5, kcps, kcar, kmod, 1
outs asig, asig
endin
</CsInstruments>
<CsScore>
f 1 0 16384 10 1
i 1 1.0 15.0 2.4321
e
</CsScore>
</CsoundSynthesizer>
```

</details>

**variation_3**

[▶ Play](https://charliesl1.github.io/CStore/#variation_3)

<details>
<summary>View CSD</summary>

```csd
<CsoundSynthesizer>
<CsOptions></CsOptions>
<CsInstruments>
seed 39595
0dbfs  = 1
instr 1
kcps = 110
kcar = 1
kmod = p4
kndx oscil 30, .25/p3, 1
kndx ceil kndx
asig foscili .5, kcps, kcar, kmod, kndx
outs asig, asig
endin
</CsInstruments>
<CsScore>
f 1 0 16384 10 1
i 1 2.5 22.5 2.9411
e
</CsScore>
</CsoundSynthesizer>
```

</details>

---

## Dataset & Reproducibility

| Dataset | Value |
|---------|-------|
| Raw .csd files | 27,104 |
| Train / Val / Test | 80/10/10 (seed 42) |
| Training docs | 21,681 |
| Vocab size | 50,265 |

**Env** — Python 3.11.11, PyTorch 2.8.0, Transformers 4.56.2, Csound 6.18.0

---

## Citation

```bibtex
@inproceedings{shi2025cstore,
  title={CStore: Reframing Text-to-Music from ``One-Shot Waveforms'' to an Interpretable, Durable, and Editable Csound Specification},
  author={Shi, Li (Charlie) and Boulanger, Richard},
  booktitle={IEEE CAI Conference},
  year={2026}
}
```

---

## Project Structure

```
CStore/
├── Convertor/     # Cabbage/Csound tools
├── docs/          # GitHub Pages — audio player (charliesl1.github.io/CStore)
├── model/         # Training, evaluation, generation
│   ├── checkpoints/   # V1.0.0, V1.0.1, V1.0.2
│   ├── evaluate.py
│   └── generate.py
├── ref output/    # Audio samples (.csd + .mp3)
└── media/
```

---

## Authors & Contact

**Li (Charlie) Shi** — [cshi@berklee.edu](mailto:cshi@berklee.edu)  
**Richard Boulanger** — [rboulanger@berklee.edu](mailto:rboulanger@berklee.edu)

Project: [github.com/CharlieSL1/CStore](https://github.com/CharlieSL1/CStore)

Thanks to the Csound community and **Salman Aslam** for model development support.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

[contributors-shield]: https://img.shields.io/github/contributors/CharlieSL1/CStore.svg?style=for-the-badge
[contributors-url]: https://github.com/CharlieSL1/CStore/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/CharlieSL1/CStore.svg?style=for-the-badge
[forks-url]: https://github.com/CharlieSL1/CStore/network/members
[stars-shield]: https://img.shields.io/github/stars/CharlieSL1/CStore.svg?style=for-the-badge
[stars-url]: https://github.com/CharlieSL1/CStore/stargazers
[issues-shield]: https://img.shields.io/github/issues/CharlieSL1/CStore.svg?style=for-the-badge
[issues-url]: https://github.com/CharlieSL1/CStore/issues
[license-shield]: https://img.shields.io/github/license/CharlieSL1/CStore.svg?style=for-the-badge
[license-url]: https://github.com/CharlieSL1/CStore/blob/main/LICENSE
