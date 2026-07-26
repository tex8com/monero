# Wallet-CUDA-Testbench

Diese Testbench misst ausschließlich den isolierten Wallet-Kryptokern

```text
D = (8 * Scalar::from_bytes_mod_order(a)) * R
```

Sie verarbeitet deterministische, öffentliche `MWMTV1`-Vektoren. Echte
Wallet-Schlüssel dürfen weder in Vektordateien noch in GPU-Puffern verwendet
werden.

## Implementierte Kandidaten

| Variante | Zweck | Register/Thread | lokaler Speicher/Thread |
|---|---|---:|---:|
| C0 `layout` | Dispatch- und Layoutobergrenze, keine Kryptografie | 22 | 0 B |
| C1 `ladder` | konstante 256-Bit-Ladder, Additionsketten | 255 | 80 B |
| C2 `radix16` | Radix-16 und 8er-Niels-Tabelle im 16-Bit-Feldkern | 255 | 9.616 B |
| C3 `radix2625-chunk` | 10-Limb-Feldkern und dreistufige Chunk-Inversion | 255 | 952 B |
| C4 `radix2625-direct` | C3-Feldkern als Einpass-Kernel | 255 | 952 B |
| C5 `radix2625-radix8` | 4er-Niels-Tabelle, signierte Radix-8-Ziffern | 255 | 176 B |
| C6 `radix2625-radix8-sqrt-ratio` | C5 mit kombinierter Dalek-`sqrt_ratio_i`-Punktdekompression | 255 | 176 B |

Die Werte stammen aus `ptxas` für `sm_86`. C6 ist der aktuelle Kandidat für
die RTX 3090. C5s kleinere Tabelle kostet mehr Punktadditionen, reduziert aber
die für CUDA besonders teuren Spill-Zugriffe deutlich. C6 vermeidet darüber
hinaus eine getrennte Inversion und Quadratwurzel beim Dekomprimieren von `R`.

## Lokale Byteprüfung

Der portable Referenzlauf benötigt kein CUDA und prüft dieselben Ergebnisbytes
und denselben ungültigen Punkt wie der GPU-Lauf:

```sh
clang++ -O3 -std=c++17 -Wall -Wextra -Werror \
  reference.cpp -o wallet-cuda-reference

./wallet-cuda-reference \
  --vectors /path/to/vectors.mwmtv1 \
  --variant all
```

## SCP- und Build-Ablauf

Der Quelltext wird lokal geändert und anschließend kopiert. Auf der
GPU-Instanz wird kein Quelltext angelegt oder editiert:

```sh
scp derivation_core.cuh derivation_radix2625.cuh vector_corpus.hpp \
  main.cu build.sh GPU:/workspace/monero-cuda/src/

ssh GPU \
  'CUDA_ARCH=sm_86 /workspace/monero-cuda/src/build.sh \
   /workspace/monero-cuda/wallet-cuda-testbench'
```

Beispiel für den bestätigten C5-Lauf:

```sh
./wallet-cuda-testbench \
  --vectors vectors/points-8192.mwmtv1 \
  --variant radix2625-radix8 \
  --rounds 60 \
  --warmup-rounds 3 \
  --threads-per-block 32
```

## RTX-3090-Ergebnisse

System: Vast-Instanz `45848910`, RTX 3090 mit 24 GB, Compute Capability 8.6,
CUDA 12.8, Treiber 595.71.05 und einem vom Host gesetzten 300-W-Limit. Alle
Tabellenwerte sind der Median aus drei formalen Läufen und enthalten nur die
GPU-Event-Zeit der Kernelpipeline.

| Punkte/Charge | C1 Ladder | C2 Radix-16/16-Bit | C3 Chunk | C4 direkt | C5 Radix-8 |
|---:|---:|---:|---:|---:|---:|
| 8 | 598 | 862 | 3.586 | 3.584 | 5.358 |
| 64 | 4.790 | 6.886 | 28.660 | 28.671 | 42.869 |
| 1.024 | 78.540 | 101.987 | 471.261 | 467.071 | 657.714 |
| 8.192 | 627.866 | 418.892 | 3.606.365 | 3.639.870 | **5.309.030** |
| 131.072 | 1.428.556 | 729.860 | 4.143.638 | 4.171.474 | **9.534.593** |

C0 ist keine Kryptografie und wird deshalb nicht als Beschleunigungswert
verwendet. Bei 8.192 Punkten erreicht C5 gegenüber dem Median der
Single-Thread-Original-Ref10-Messung (`27.030,166/s`) den Faktor
**196,41×**. Gegenüber C1 beträgt der Faktor **8,456×**.

Der zusätzliche Dauerlauf verarbeitete 131.072.000 Ableitungen in
13,683650391 Sekunden:

```text
9.578.730,548 Ableitungen/s
Temperatur: 52–69 °C
Leistung:   maximal 299,55 W
SM-Takt:    maximal 1.740 MHz
validation=pass
```

## RTX-3090-C6-Ergebnisse

System: Vast-Instanz `45854234`, RTX 3090 mit 24 GB, Compute Capability 8.6,
CUDA 12.8, Treiber 590.48.01 und 350-W-Limit. C5 und C6 wurden alternierend
auf demselben Host mit jeweils fünf formalen Läufen und 128 Threads pro Block
vermessen. Die Tabelle enthält den Median der GPU-Event-Zeit.

| Punkte/Charge | C5 Radix-8 | C6 `sqrt_ratio_i` | C6/C5 |
|---:|---:|---:|---:|
| 8.192 | 6.111.302 | **6.462.993** | **1,0575×** |
| 131.072 | 10.331.329 | **10.948.124** | **1,0597×** |

Gegenüber der Single-Thread-Original-Ref10-Messung (`27.030,166/s`) erreicht
C6 bei 131.072 Punkten den Faktor **405,03×**.

Der C6-Dauerlauf verarbeitete 262.144.000 Ableitungen in 24,294734375
Sekunden:

```text
10.790.157,075 Ableitungen/s
GPU-Auslastung:    100 % in allen aktiven Telemetrie-Samples
Speicherauslastung:  0 %
Leistung:          Median 348,82 W, maximal 349,27 W bei 350 W Limit
Temperatur:        51–65 °C
SM-Takt:           Median 1.830 MHz, 1.815–1.845 MHz
validation=pass
```

C5 und C6 benötigen beide 255 Register und 176 B lokalen Speicher pro Thread.
Bei 128 Threads pro Block passen deshalb statisch zwei Blöcke beziehungsweise
acht Warps auf jeden Ampere-SM: 16,7 % theoretische Warp-Occupancy. Die
Telemetrie zeigt gleichzeitig 100 % GPU- und 0 % Speicherauslastung; der
aktuelle Kernel ist damit Compute-, Register- und Power-limitiert, nicht durch
globale Speicherbandbreite.

## Prüfstatus

- Bytegleich gegen Curve25519-Dalek 4.1.3: bestanden.
- Dalek-verworfener Punkt: `valid=0` und 32 Nullbytes, bestanden.
- C5-Skalarrekodierung: Null, Eins, Gruppenordnung, `ff…ff` und 10.000
  deterministische 256-Bit-Eingaben rekonstruiert, bestanden.
- C6-`sqrt_ratio_i`: portable Byteprüfung gegen Curve25519-Dalek 4.1.3 für
  1.024 Punkte sowie GPU-Prüfungen für 8 bis 131.072 Punkte, bestanden.
- Drei formale Wiederholungen je Variante und Corpusgröße: bestanden.
- CUDA Compute Sanitizer `memcheck`: alle C0–C5-Pfade ohne Fehler.
- `initcheck`: C3–C5 ohne Fehler.
- C6 `memcheck` und `initcheck`: jeweils 0 Fehler.
- `synccheck` und `racecheck`: C1/C2 ohne Fehler beziehungsweise Hazards.
- Blockgrößen 32, 64, 128 und 256: vermessen.
- C3-Chunkgrößen 1, 2, 4, 8, 16, 32, 64 und 128: vermessen.
- Registergrenze 192 gegen den natürlichen 255-Register-Build: vermessen und
  verworfen.

Nsight Compute konnte auf dem Mietsystem keine Performance-Counter lesen
(`ERR_NVGPUCTRPERM`). Die Ressourcenwerte stammen deshalb aus
`nvcc -Xptxas=-v`; Timing erfolgt mit CUDA Events.

Die vollständigen Rohdaten liegen unter
`build/wallet-cuda-testbench/rtx3090-vast-45848910-20260725/` (C0–C5) und
`build/wallet-cuda-testbench/rtx3090-c6-vast-45854234-20260725/` (C5/C6).

Die vollständige Auswertung der nachfolgenden C7–C11-Experimente auf der
RTX 3090 sowie aller SM-120-Messungen auf der RTX 5090 steht in
[`RESULTS-2026-07-25.md`](RESULTS-2026-07-25.md).
