# civilocity-redline

Phase 1 Redline Checker — source code repository.

This repository contains **source code only**. Authority documents,
audit data, sample files, and generated outputs live in the parent
project directory and are never committed here.

## Project structure

This repository is the `src\` subdirectory of the larger civilocity_reviewer
project. Authority documents and data are at sibling paths on disk but are
not visible to git:

```
civilocity_reviewer\          (project root, NOT a git repo)
├── audit\                    (RC_Audit_v1.xlsx, manifest)
├── decisions\                (RC_Decisions_Log_v1.docx)
├── handoff\                  (active and archived session notes)
├── outputs\                  (generated checklists, eval results)
├── reference\                (RC_Reference_Document_v1_FINAL.docx)
├── samples\                  (development and holdout PDFs/PNGs — never committed)
├── src\                      ← THIS REPOSITORY (civilocity-redline on GitHub)
└── stack\                    (RC_Stack_and_Environment_Brief_v1.docx)
```

## Development

Python 3.11. Setup is defined by `pyproject.toml` once Build Step 2 completes.

## Do not commit

- Audit workbooks (`*.xlsx`)
- Redline sample files (any PDF, PNG, JPG in `samples\`)
- Holdout files
- Generated JSON outputs (manifests, extraction results)
- Generated checklist PDFs
- Client project data
- Authority Word documents (`*.docx`)

The `.gitignore` enforces these rules; the list above explains *why*.

## Authority

The five governing documents for this project live at the project root,
not in this repository:

1. `RC_Reference_Document_v1_FINAL.docx` — controlling specification
2. `RC_Audit_v1.xlsx` — corpus audit and ground truth
3. `RC_Decisions_Log_v1.docx` — binding decisions D-001 through D-031
4. `RC_Stack_and_Environment_Brief_v1.docx` — environment and source rules
5. `RC_Session_Handoff_Note_v1.docx` — current session state

Read those before proposing changes to this codebase.
