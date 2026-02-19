<CsoundSynthesizer>
<CsOptions>
</CsOptions>
<CsInstruments>

0dbfs  = 1

instr 1

kfreq = 220
kc1 = 5
kc2 = p4
kvrate = 6

kvdpth line 0, p3, p5
asig   fmb3 .4, kfreq, kc1, kc2, kvdpth, kvrate
      outs asig, asig

endin
</CsInstruments>
<CsScore>
; sine wave.
f 1 0 32768 10 1

i 1 0 2  5 0.1
i 1 3 2 .5 .5 0.01
i 1 4 .5 .5 0.01
e
</CsScore>
</CsoundSynthesizer>
