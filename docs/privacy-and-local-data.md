# Privacy and Local Data Guidance

CornerCrop is designed for local image-library processing. Public repository
content should describe reusable workflows without exposing private collections
or machine details.

## Do Not Commit

- Input image libraries or generated crops.
- SQLite run databases, WAL/SHM files, logs, TSV review exports, or progress
  snapshots.
- Local mount points, usernames, NAS addresses, or exact private directory
  structures.
- Private run notes containing collection names, corpus sizes, or unresolved
  manual-review paths.
- Credentials, API keys, SSH keys, cookies, or service tokens.

## Safe Places for Local Notes

These paths are ignored and may be used for private local work:

- `runs/`
- `inputs/`
- `outputs/`
- `reports/`
- `audits/`
- `docs/private/`
- `docs/local/`
- `*.local.md`

## Before Publishing

Run:

```bash
git status --short
rg -n -i "(/Volumes|/Users|192\\.168|api[_-]?key|secret|token|password|bearer)" \
  README.md docs scripts src tests pyproject.toml .gitignore
```

For history checks:

```bash
git log --all -G "(/Volumes|/Users|192\\.168|api[_-]?key|secret|token|password|bearer)" -- .
git log --all --name-only --pretty=format: | sort -u
```

If a real credential ever enters Git history, rotate it immediately and rewrite
history before publishing the repository.
