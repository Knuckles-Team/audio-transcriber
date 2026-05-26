# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.0] - 2026-05-22

### Added
- Standard environment variables section inside the repository-wide `README.md`.
- Default environment configuration keys in `.env.example`.
- Concept traceability maps across all key components and test files referencing `agent-utilities` core features (`OS-5.4`, `OS-5.1`, `OS-5.3`, `ORCH-1.4`, `OS-5.2`).
- Structured `CHANGELOG.md` following the Keep a Changelog standard format.

### Changed
- Improved test coverage to ensure solid execution and structural guarantees under Pytest.
- Unified import filters for FastMCP and Requests warnings to improve system startup output.
