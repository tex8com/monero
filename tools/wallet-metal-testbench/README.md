# Wallet-Metal-Testbench

Dies ist die GPU-Teststrecke für den Wallet-Scan. Sie enthält keine
Produktintegration und verarbeitet ausschließlich deterministische, öffentliche
Testdaten. Ein echter View Key darf weder in den Testvektor noch in den GPU-Puffer.

## M0 – Layout und Dispatch (historische Baseline)

M0 bindet genau die später benötigten Puffer:

```text
View-Key: 32 Byte
R:        N × 32 Byte
D:        N × 32 Byte
valid:    N × 1 Byte
```

Der Kernel kopiert nur `R` bytegleich nach `D` und setzt `valid`. Er misst
Metal-Initialisierung, Unified-Memory-Übergabe, Dispatch und Ergebnisabnahme.
**M0 enthält keine Edwards25519-Arithmetik; seine Werte sind keine
Ableitungen/s und dürfen nicht mit CPU-Kryptowerten verglichen werden.**

## M1 – vollständige Referenzableitung

`derivation.metal` implementiert die gleiche Operation wie der aktuelle
Rust-Adapter:

```text
D = (8 * Scalar::from_bytes_mod_order(a)) * R
```

Der Kernel enthält Feldarithmetik modulo `2^255 - 19`, vollständige
Extended-Edwards-Formeln, Punktdekodierung, Skalarreduktion modulo der
Ed25519-Gruppenordnung und komprimierte Ausgabe. Sein Ziel ist zunächst
Korrektheit, nicht Höchstleistung.

Vor jeder M1-Messung baut
`tools/wallet-crypto-testbench` eine binäre `MWMTV1`-Vektordatei. Sie enthält
deterministische, öffentliche Punkte und für jeden Punkt das Byte-genaue
Ergebnis des vorhandenen Dalek-Adapters. Zusätzlich enthält sie einen von
Dalek verworfenen Punkt. Der Metal-Lauf besteht nur, wenn:

1. jedes der `N` Ergebnisse exakt den 32 Dalek-Bytes entspricht und
2. der ungültige Punkt `valid=0` und 32 Nullbytes liefert.

Beispiel für einen kleinen Korrektheitslauf:

```sh
WALLET_METAL_M1_POINTS=8 \
WALLET_METAL_BENCH_RUN_ID=metal-m1-smoke-mac-m4-YYYYMMDD-001 \
bash tools/wallet-metal-testbench/run-metal-m1-derivation-testbench.sh \
  --rounds 1 --warmup-rounds 1 --threads-per-group 32
```

Der Vektorexport ist eine Korrektheitsvoraussetzung, keine CPU-Performance-
Messung. `m1_derivations_per_second` darf erst nach `validation=pass` mit der
CPU-Ableitungsrate verglichen werden; er bleibt dennoch eine isolierte
Kryptokern-Messung, keine Wallet-Sync-Rate.

## M2 – gemeinsamer Skalar pro Threadgruppe

`derivation_m2_group_scalar` verwendet die unveränderte M1-Mathematik und
dieselben `MWMTV1`-Vektoren. Der einzige Unterschied ist die Ausführung: Da
eine Wallet-Scan-Charge einen gemeinsamen View-Skalar hat, reduziert und
multipliziert eine Lane den Skalar mit 8 einmal pro Threadgruppe; die übrigen
Lanes übernehmen das 32-Byte-Ergebnis nach einer Threadgroup-Barriere.

```sh
WALLET_METAL_M1_POINTS=8192 \
WALLET_METAL_BENCH_RUN_ID=metal-m2-bulk-r1-mac-m4-YYYYMMDD-001 \
bash tools/wallet-metal-testbench/run-metal-m1-derivation-testbench.sh \
  --kernel derivation_m2_group_scalar \
  --rounds 3 --warmup-rounds 1 --threads-per-group 32
```

M2 gilt nur dann als Verbesserung, wenn dieselben Byte- und Fehlerpfadprüfungen
wie M1 bestehen. Auch M2 bleibt ein Testkern, nicht Wallet-Produktcode.

## M4 – Dalek-Radix-16 mit projektiven Niels-Tabellen

M4 übernimmt den Algorithmus der variablen Basispunkt-Multiplikation aus
`curve25519-dalek` 4.1.3, statt nur eine einzelne Feldoperation zu verändern:

1. Es erzeugt pro Eingangspunkt eine Tabelle `P, 2P, …, 8P` in
   Projektiv-Niels-Koordinaten.
2. Den bereits auf `8*a mod l` gefalteten Skalar zerlegt es in 64 signierte
   Radix-16-Ziffern (`[-8, 8]`).
3. Es verarbeitet die Ziffern von oben nach unten mit je vier Doublings in
   P2-Koordinaten und einer Niels-Tabellenaddition.

Damit sinkt die Anzahl der Punktadditionen im Multiplikationshauptpfad von 256
auf 64; der Tabellenaufbau benötigt zusätzlich sieben Additionen. Die
Punktdekodierung, Skalarvertrag, Ergebniscodierung und der M2-Threadgroup-
Skalarpfad bleiben unverändert. M4 wird erst nach vollständiger Byteprüfung
gegen dieselben `MWMTV1`-Dalek-Vektoren als Kandidat betrachtet.

```sh
WALLET_METAL_M1_POINTS=8192 \
WALLET_METAL_BENCH_RUN_ID=metal-m4-radix16-r1-mac-m4-YYYYMMDD-001 \
bash tools/wallet-metal-testbench/run-metal-m1-derivation-testbench.sh \
  --kernel derivation_m4_radix16_niels \
  --rounds 3 --warmup-rounds 1 --threads-per-group 32
```

## M5 – M4 plus Dalek-Feld-Additionsketten

M5 behält den vollständigen M4-Skalarmultiplikationspfad bei, ersetzt jedoch
die generische Binärexponentiation beim Dekodieren und Kodieren der Punkte:

- Feldinversion `x^(p-2)`: dieselbe `pow22501`-Additionskette wie Dalek,
  254 Quadrierungen und 11 volle Multiplikationen;
- Quadratwurzel `x^((p+3)/8)`: dieselbe Kette, 252 Quadrierungen und 11 volle
  Multiplikationen.

M4 führt im Referenzfeldkern für diese Exponenten noch eine volle
Multiplikation für fast jedes gesetzte Exponentbit aus. M5 ändert **nicht** die
Feldrepräsentation, die Reduktion oder die Punktformeln; damit ist die
Auditoberfläche gegenüber einem neuen Feldkern klein. Der obligatorische
Dalek-Byte- und Ungültig-Punkt-Test bleibt unverändert.

```sh
WALLET_METAL_M1_POINTS=8192 \
WALLET_METAL_BENCH_RUN_ID=metal-m5-addition-chain-r1-mac-m4-YYYYMMDD-001 \
bash tools/wallet-metal-testbench/run-metal-m1-derivation-testbench.sh \
  --kernel derivation_m5_radix16_niels_addition_chain \
  --rounds 3 --warmup-rounds 1 --threads-per-group 32
```

## M12/M16 – 25/26-Bit-Feldkern und blockweise Batch-Inversion

`derivation_radix2625_chunkinvert.metal` ist der aktuelle Metal-Kandidat. Er
behält die Radix-16-Punktmultiplikation und Dalek-Additionsketten bei, verwendet
aber Daleks unsigned 10-Limb-Feldrepräsentation. Die Ausgabe läuft in drei
geordneten Metal-Pässen:

1. Punktdekodierung und Skalarmultiplikation in Projektivkoordinaten;
2. Montgomery-Batch-Inversion in unabhängigen 16-Punkte-Chunks;
3. parallele affine Konvertierung und komprimierte Edwards-Ausgabe.

Der Host verwendet für jeden In-Flight-Slot getrennte Projektiv-, Inversen-,
Ergebnis- und Gültigkeitspuffer. Der Fehlerpfad und der vollständige
Dalek-Byteabgleich gelten unverändert.

```sh
WALLET_METAL_M1_POINTS=8192 \
WALLET_METAL_BENCH_RUN_ID=metal-m16-pass-groups-r1-mac-m4-YYYYMMDD-001 \
WALLET_METAL_KERNEL_SOURCE="$PWD/tools/wallet-metal-testbench/derivation_radix2625_chunkinvert.metal" \
bash tools/wallet-metal-testbench/run-metal-m1-derivation-testbench.sh \
  --kernel derivation_m12_projective_chunkinvert \
  --rounds 60 --warmup-rounds 3 \
  --threads-per-group 64 \
  --projective-threads-per-group 256 \
  --inverse-threads-per-group 16 \
  --compress-threads-per-group 128 \
  --batch-inversion-chunk-size 16
```

M16 verwendet denselben M12-Kern und dieselbe Mathematik. Nur die
Threadgruppengröße wird für die drei Pässe getrennt gewählt. Die oben
angegebenen Werte sind die auf einem Apple M4 formal bestätigte Einstellung.
Andere GPUs müssen mit der öffentlichen Testbench separat abgestimmt werden;
`--threads-per-group` bleibt der gemeinsame Rückfallwert für Projektiv- und
Kompressionspass.

Die übrigen Entwicklungsstufen wurden getrennt vermessen und verworfen
beziehungsweise in M12 übernommen:

- M7: erster 25/26-Bit-Port;
- M8: verzögerter Carry bei Additionen;
- M9: kanonische Limb-Vergleiche;
- M10: gemeinsam kodierte Skalarziffern;
- M11: korrekte, aber langsame serielle Batch-Inversion;
- M13: korrekte, aber langsame reine 32-Bit-Radix-`2^13`-Arithmetik.
- M14: korrekter, aber langsamerer paralleler Threadgroup-Scan;
- M15: korrekter, aber langsamerer 32-Lane-SIMD-Scan.

Keine dieser Dateien ist Produktintegration. Der Testbench darf weiterhin
keinen echten Wallet-Schlüssel verarbeiten.
