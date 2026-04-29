# Contributing to CornerCrop

Thanks for helping improve CornerCrop.

## Development Setup

CornerCrop is macOS-only because it uses Apple Vision.framework through PyObjC.

```bash
python -m pip install -e ".[dev]"
uv run --extra dev pytest
```

The test suite is intentionally small and should stay fast. Add focused tests for
crop policy, verification behavior, and large-run resume semantics when changing
those areas.

## Local Data Hygiene

Do not commit private image libraries, generated crops, run databases, audit
exports, machine-local paths, or private notes. The repository ignores the common
local locations (`runs/`, `inputs/`, `outputs/`, `docs/private/`,
`docs/local/`, SQLite/WAL files, logs, and TSV exports), but please run a quick
scan before opening a pull request:

```bash
git status --short
rg -n -i "(/Volumes|/Users|192\\.168|api[_-]?key|secret|token|password|bearer)" \
  README.md docs scripts src tests pyproject.toml .gitignore
```

If you need to document a production run, keep private paths and exact corpus
details in ignored local notes. Public docs should describe reusable operating
patterns rather than private collection details.

## Pull Request Checklist

- Tests pass with `uv run --extra dev pytest`.
- Public docs avoid private paths, collection names, local usernames, and exact
  private run statistics.
- New long-running workflows have resumable state and a clear audit path.
- Destructive file operations are guarded by dry-run or explicit user action.
