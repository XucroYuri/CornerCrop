# Changelog

All notable changes to CornerCrop will be documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning once releases are tagged.

## Unreleased

### Added

- Resumable large-library runner with local SQLite state, archive handling, and
  resource-aware album parallelism.
- Second-pass recovery for non-corner watermark archives.
- Heartbeat progress for long in-flight OCR work.
- Generic large-library runbook, privacy guidance, and helper scripts.
- GitHub Actions CI for macOS.

### Changed

- Second-pass recovery now evaluates all viable crop profiles and prefers the
  lowest-loss verified-clean crop.
- Public documentation now avoids private paths, collection names, and exact
  private run statistics.
