# IRCN Development Completion Gate v1

**Effective from:** 2026-10-08
**Purpose:** prevent code from being presented as task-complete before its implementation has been checked for defects and its verification evidence has been reviewed.

## Required order

Every code-bearing task follows these gates in order. “Implementation finished” means the author believes the requested change is present; it is not yet a completion claim.

### Gate 0 — Freeze scope and acceptance contract

Before implementation, record the expected behavior, affected paths, acceptance checks, compatibility constraints, and any scientific or data boundary. If the task changes a frozen experiment contract, stop and record a protocol amendment before using confirmatory data. Do not silently tune acceptance criteria after seeing test results.

### Gate 1 — Implement and self-review

After coding, inspect the complete diff and trace changed behavior through callers, state transitions, errors, boundary cases, and resource paths. Check for the specific defect classes relevant to the task: incorrect assumptions, off-by-one/state-order errors, stale or duplicated events, input validation gaps, numerical instability, data leakage, missing cost accounting, and incomplete logging. Record the exact commit or diff reviewed.

### Gate 2 — Defect review before completion packaging

Review the implementation against the acceptance contract before writing a completion report, marking a work package complete, publishing results as complete, or requesting merge. Use an independent reviewer when available. Otherwise perform a separate read-only review pass after implementation and disclose that the reviewer was not independent.

Record findings by severity:

- **Blocker:** wrong scientific estimand, invalid data/provenance, causal/state corruption, security/data loss, or result cannot be reproduced. Must be fixed; no downstream run or completion package.
- **Major:** acceptance behavior is wrong, important edge cases fail, comparison is unfair, or required cost/evidence is omitted. Must be fixed and re-reviewed.
- **Minor:** non-blocking defect with bounded effect. Fix before completion when practical; otherwise document scope, impact, and follow-up owner. A minor finding cannot conceal a failed acceptance check.
- **Observation:** maintainability or future-risk note without demonstrated current failure. Record if material; it does not by itself block completion.

Do not describe code as “bug-free.” A review can establish only that no known blocking or major defects remain within the reviewed scope and evidence. Any unresolved uncertainty must be stated.

### Gate 3 — Verify fixes and acceptance

For each blocker/major finding, add or identify a regression check, fix it, rerun the relevant checks, then re-review the affected code. A fix reopens Gate 2 for that changed behavior. Run the narrowest tests that prove the changed behavior, then the repository-required regression suite, static checks, and reproducibility/hash checks where applicable. Do not report a check as passed unless its fresh output was inspected.

If a required check fails, is skipped, or cannot run, the task remains **NOT READY**. Preserve failure output and state the reason; do not replace a failed test with an unsupported narrative claim.

### Gate 4 — Prepare completion package

Only after Gates 0–3 pass, prepare the completion package. It must include:

1. changed paths and behavior;
2. review scope, findings, fixes, and any non-independent review limitation;
3. exact verification commands and pass/fail output;
4. known limitations and unresolved minor findings;
5. commit/branch, provenance hashes, and raw logs for experiment-bearing work;
6. a status that distinguishes code verification from scientific evidence.

For experiments, software correctness does not imply scientific gate passage. Preregistration, task quality, fair baselines, sample unit, uncertainty, and stopping rules remain separate acceptance items.

### Gate 5 — Publish and merge

Stage only the approved `model/` and `result/` paths for research-stage artifacts unless the user explicitly changes that repository boundary. Verify the PR path list and rerun required checks on the final PR head. Merge only after the final diff is reviewable and Gates 0–4 are documented. Then verify the merge commit and update the corresponding Feishu stage record.

## Completion record template

Copy this for each code-bearing stage:

```text
Status: READY | NOT READY
Acceptance contract:
Reviewed diff/commit:
Review mode: independent reviewer | separate self-review (not independent)
Findings: blocker / major / minor / observation, with file and evidence
Fixes and re-review:
Targeted checks:
Regression/static/reproducibility checks:
Known limitations:
Scientific status (if applicable):
PR paths and merge commit:
Feishu revision:
```

## Stop condition

The task may be marked complete only when Gate 4 is satisfied. If a blocker or major defect remains, label the work **NOT READY**, preserve the evidence, and return to implementation/review. “No known defects found” is a scoped evidence statement, not proof that no bug exists.
