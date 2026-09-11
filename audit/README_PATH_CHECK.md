# README path check

Checked on 2026-09-11.

## Scope

- `README.md`
- `docs/data_access.md`
- README-linked command and documentation paths

## Result

PASS: public workflow documentation contains no machine-specific home-directory paths. Commands use repository-relative paths, environment variables, and `../project-assets` placeholders for user-owned data.

Historical source-provenance reports under `audit/` may retain original absolute paths as traceability records; those are not executable README instructions and were not rewritten in this documentation repair.

The public `figures/` directory and generated picture files remain removed by design.
