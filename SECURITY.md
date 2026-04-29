# Security Policy

## Supported Versions

CornerCrop is currently pre-1.0. Security fixes are made on the default branch.

## Reporting a Vulnerability

Please do not open a public issue for sensitive reports.

Email the maintainer listed on the GitHub repository profile, or use GitHub's
private vulnerability reporting feature if it is enabled for the repository.
Include:

- A short description of the issue.
- A minimal reproduction when possible.
- Whether private files, paths, or credentials may be exposed.

## Data Safety Scope

CornerCrop processes local image files and can overwrite or move files during
large in-place runs. Treat image libraries, run databases, reports, and review
exports as private unless you have explicitly sanitized them.

The repository is configured to ignore common local artifacts, but contributors
are responsible for checking that no private data is staged before publishing.
