# Run data

Everything the experiments produced, so results can be inspected and the dashboard run without re-flying anything.

| Path | What |
|---|---|
| `drone_jev.db.gz` | The main SQLite database (all experiments, scenarios, arms, flights, decisions, budget ledger). **Compressed** because GitHub rejects files over 100 MB. |
| `demo.db`, `demo/` | A small seeded database and its trajectories, used by the docs and the dashboard demo |
| `trajectories/<experiment_id>/<flight_id>.parquet` | One Parquet file per flight, one row per 48 Hz control tick |
| `orb_game/` | Orb Dodge flights (`.npz` trajectories and `.json` metadata), and the tournament |
| `calibration.json` | The obstacle-avoidance calibration table used by the scenario generator |

## Unpack the main database

```bash
gunzip -k data/drone_jev.db.gz     # creates data/drone_jev.db and keeps the .gz
```

`data/drone_jev.db` is gitignored (only the compressed copy is committed). The API, the dashboard and the scripts read `data/drone_jev.db`, so unpack it first. To refresh the committed copy after new runs:

```bash
sqlite3 data/drone_jev.db "PRAGMA wal_checkpoint(TRUNCATE)"
gzip -9 -n -k -f data/drone_jev.db
```
