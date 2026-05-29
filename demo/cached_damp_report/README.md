# Cached DAMP Report Demo

Open `index.html` in a browser and click **Start Demo**.

This is a cached visual demo for presentation recording. It does not run GPU inference; it replays the final pipeline stages and reveals a precomputed qualitative output from `damp_full_qualitative_100.zip`.

Selected output image:

```text
train_000066_damp_full.png
```

The metric cards use the current downstream summary:

| Method | mIoU |
|---|---:|
| Zero-shot | 0.1310 |
| DAMP prompt-only | 0.1316 |
| DAMP full | 0.1302 |
