# Historischer Monero-Ref10-Testbench

Dieser Testbench misst nur die historische Implementierung von
`crypto_ops::generate_key_derivation`, wie sie unmittelbar vor dem Fast-Crypto-
Commit `e0a84c6e8` aussah:

```text
ge_frombytes_vartime → ge_scalarmult → ge_mul8 → ge_tobytes
```

Er verwendet ausschließlich `MWMTV1`-Vektoren des Rust-Testbenchs: einen
deterministischen öffentlichen kanonischen Skalar, öffentliche Punkte und die
jeweils erwartete Dalek-Ausgabe. Vor dem Timer prüft er jeden Punkt Byte für
Byte gegen Dalek und verifiziert, dass ein ungültiger Punkt verworfen wird.

Beispiel für eine serielle Original-Referenz:

```sh
WALLET_ORIGINAL_REF10_WORKERS=1 \
WALLET_ORIGINAL_REF10_RUN_ID=original-ref10-serial-r1-mac-m4-YYYYMMDD-001 \
bash tools/wallet-original-crypto-testbench/run-original-ref10-benchmark.sh
```

`workers=10` misst denselben historischen Kern mit einer bewusst externen,
persistent arbeitenden Benchmark-Parallelisierung. Das ist keine Behauptung,
dass die historische Wallet diesen Parallelisierungsgrad besaß; die serielle
Messung bleibt die Verhaltensreferenz der ursprünglichen Funktion.
