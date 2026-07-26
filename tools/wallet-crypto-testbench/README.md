# Wallet-Krypto-Testbench

Dieser Testbench misst ausschließlich die Monero-Key-Derivation `D = 8 * a * R`.
Er enthält weder Node-Transport, Datenbankzugriffe, ScanPack, Blockparser noch
Wallet-Commit. Damit dürfen seine Derivations/s niemals mit Blöcken/s,
Payload-MiB/s oder vollständiger Wallet-Syncdauer gleichgesetzt werden.

## Varianten

- `dalek_direct_parallel`: Direkter `curve25519-dalek`-Aufruf mit dem
  ursprünglichen Algorithmus.
- `wallet_scalar_ffi_parallel`: Aktueller Produktadapter
  `fast_generate_key_derivation` unter demselben Worker-Budget.

Vor jeder Zeitmessung validiert der Testbench für alle Punkte die Bytegleichheit
der beiden Wege sowie den Fehlerpfad für einen ungültig kodierten Punkt. Der
Korpus besteht aus deterministisch erzeugten, gültigen Edwards25519-Punkten;
Korpusaufbau und Prüfung liegen außerhalb der Zeitmessung. Das simuliert die
Form der Transaktions-Public-Keys im Wallet, ersetzt aber keinen vollständigen
Wallet-Sync-Test.

## Ausführung

```sh
tools/wallet-crypto-testbench/run-derivation-benchmark.sh \
  --workers 10 --points 131072 --rounds 100 --warmup-rounds 2 --variant all
```

Jeder Lauf erzeugt einen neuen Ergebnisordner unter
`build/wallet-crypto-testbench/` mit Build-Log, vollständiger Umgebung,
Quell-Checksummen, Rohwerten und `/usr/bin/time`-Ressourcenwerten.
