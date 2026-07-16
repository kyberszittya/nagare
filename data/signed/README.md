# Signed-graph datasets (not committed — download on demand)

Real signed graphs for the Gömb-Soma signed-link predictor (`examples/cpml_signed_link.rs`).
The raw CSVs are gitignored; fetch with:

```
curl -sSL https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz | gunzip > data/signed/bitcoinalpha.csv
# (optional, larger)
curl -sSL https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz  | gunzip > data/signed/bitcoinotc.csv
```

Format: `SOURCE,TARGET,RATING,TIME` (the loader reads the first 3 cols; TIME ignored).
Source: SNAP soc-sign-bitcoin (Kumar et al.). bitcoin-alpha: V=3783, 24186 signed edges.
