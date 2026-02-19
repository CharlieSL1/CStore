<CsoundSynthesizer> 
<CsOptions>
</CsOptions>
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
