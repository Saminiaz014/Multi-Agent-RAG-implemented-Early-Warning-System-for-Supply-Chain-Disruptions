# CLAUDE.md

## Skills usage

Project-scoped plugins (see `.claude/settings.json`): **superpowers@claude-plugins-official** (v7.0.2) and **academic-research-skills@academic-research-skills** (ARS, v3.24.4).

- **Any bug or failing test** → `superpowers:systematic-debugging`. Root cause first. After 3 failed fixes, stop and report.
- **Any new code** → `superpowers:test-driven-development` (red → green → refactor).
- **Any multi-step change** → `superpowers:brainstorming`, then `superpowers:writing-plans` / `superpowers:executing-plans` with checkpoints.
- **Before any commit** → `superpowers:requesting-code-review` (dispatches a general-purpose reviewer subagent with the skill's `code-reviewer.md` template; Superpowers v7 ships no separate code-reviewer agent).
- **Paper or thesis text** → `academic-research-skills:academic-paper` (revision and citation-check modes). **Methodology review** → `academic-research-skills:academic-paper-reviewer` in methodology mode.
- **ARS output is a draft.** Never write it straight into `.tex` files.
