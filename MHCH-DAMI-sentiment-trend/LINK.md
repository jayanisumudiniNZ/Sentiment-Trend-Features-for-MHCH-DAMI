# Link to MHCH-DAMI (main)

This project **does not fork** the upstream training code. It imports the original repo at runtime.

| Resource | Location |
|----------|----------|
| Upstream code | `../MHCH-DAMI` (or `$MHCH_DAMI_ROOT`) |
| Shared datasets | `$MHCH_DAMI_ROOT/data/{clothing,makeup}/` |
| Data configs | `$MHCH_DAMI_ROOT/config/data/` |
| Path A weights | `./weights/` (this repo) |
| Path A results | `./results/` (this repo) |

Set the link:

```bash
export MHCH_DAMI_ROOT=/absolute/path/to/MHCH-DAMI
```

Or copy `.env.example` to `.env` and load it in your shell.
