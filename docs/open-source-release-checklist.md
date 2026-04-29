# Open Source Release Checklist

Use this checklist before pushing public release branches or tags.

## Repository Metadata

- `README.md` explains install, usage, limitations, development, and support.
- `LICENSE` is present.
- `CONTRIBUTING.md` is present.
- `SECURITY.md` is present.
- `CHANGELOG.md` has an `Unreleased` section.
- GitHub Actions CI passes on macOS.

## Privacy Review

- No private image paths, usernames, LAN addresses, or collection names in public docs.
- No exact private run statistics in public docs.
- No generated review exports or run databases staged.
- No credentials or service tokens in current files.
- Git history has been checked before public push.

## Functional Review

- `uv run --extra dev pytest` passes.
- Main CLIs print help successfully.
- Large-library workflows support dry-run, resume state, stop files, and audit.
- File-moving operations are collision-safe and only delete archive sources after
  verified outputs are written.
