---
name: verify-implementation
description: Sequentially executes all verify skills in the project to generate an integrated verification report. Use after feature implementation, before PRs, or during code review.
disable-model-invocation: true
argument-hint: "[optional: specific verify skill name]"
---

# Implementation Verification

## Purpose

Performs integrated verification by sequentially executing all `verify-*` skills registered in the project:

- Runs checks defined in each skill's Workflow
- References each skill's Exceptions to prevent false positives
- Suggests fixes for discovered issues
- Applies fixes and re-verifies after user approval

## When to Run

- After implementing a new feature
- Before creating a Pull Request
- During code review
- When auditing codebase rule compliance

## Skills to Execute

List of verification skills that this skill executes sequentially. `/manage-skills` automatically updates this list when creating/deleting skills.

| # | Skill | Description |
|---|-------|-------------|
| 1 | `verify-test-coverage` | Verifies new models/trainers/pipelines have corresponding unit and integration tests |
| 2 | `verify-code-convention` | Validates PEP 8 naming, type hints, ruff compliance, and import ordering |
| 3 | `verify-model-registration` | Ensures new models are registered in factory, have configs, train.py routing, and implement abstract methods |
| 4 | `verify-config-consistency` | Validates Hydra config files have required sections, correct YAML structure, and resolvable references |

## Workflow

### Step 1: Introduction

Check the skills listed in the **Skills to Execute** section above.

If an optional argument is provided, filter to that skill only.

**If there are 0 registered skills:**

```markdown
## Implementation Verification

No verification skills found. Run `/manage-skills` to create verification skills for your project.
```

In this case, terminate the workflow.

**If there is 1 or more registered skills:**

Display the contents of the Skills to Execute table:

```markdown
## Implementation Verification

The following verification skills will be executed sequentially:

| # | Skill | Description |
|---|-------|-------------|
| 1 | verify-<name1> | <description1> |
| 2 | verify-<name2> | <description2> |

Starting verification...
```

### Step 2: Sequential Execution

For each skill listed in the **Skills to Execute** table, perform the following:

#### 2a. Read Skill SKILL.md

Read the skill's `.claude/skills/verify-<name>/SKILL.md` and parse the following sections:

- **Workflow** — Check steps and detection commands to execute
- **Exceptions** — Patterns considered not a violation
- **Related Files** — List of files to check

#### 2b. Run Checks

Execute each check defined in the Workflow section in order:

1. Use the tool specified in the check (Grep, Glob, Read, Bash) to detect patterns
2. Compare detected results against the skill's PASS/FAIL criteria
3. Exempt patterns matching the Exceptions section
4. For FAIL results, record the issue:
   - File path and line number
   - Problem description
   - Recommended fix (with code example)

#### 2c. Record Per-Skill Results

After completing each skill, display progress:

```markdown
### verify-<name> Verification Complete

- Check items: N
- Passed: X
- Issues: Y
- Exempted: Z

[Moving to next skill...]
```

### Step 3: Integrated Report

After all skills have been executed, consolidate results into a single report:

```markdown
## Implementation Verification Report

### Summary

| Verification Skill | Status | Issue Count | Details |
|--------------------|--------|-------------|---------|
| verify-<name1> | PASS / X issues | N | Details... |
| verify-<name2> | PASS / X issues | N | Details... |

**Total issues found: X**
```

**When all verifications pass:**

```markdown
All verifications passed!

The implementation complies with all project rules:

- verify-<name1>: <pass summary>
- verify-<name2>: <pass summary>

Ready for code review.
```

**When issues are found:**

List each issue with file path, problem description, and recommended fix:

```markdown
### Issues Found

| # | Skill | File | Problem | Fix |
|---|-------|------|---------|-----|
| 1 | verify-<name1> | `path/to/file.ts:42` | Problem description | Fix code example |
| 2 | verify-<name2> | `path/to/file.tsx:15` | Problem description | Fix code example |
```

### Step 4: User Action Confirmation

If issues are found, use `AskUserQuestion` to confirm with the user:

```markdown
---

### Fix Options

**X issues were found. How would you like to proceed?**

1. **Fix all** - Automatically apply all recommended fixes
2. **Fix individually** - Review and apply each fix one by one
3. **Skip** - Exit without changes
```

### Step 5: Apply Fixes

Apply fixes based on the user's selection.

**If "Fix all" is selected:**

Apply all fixes in order, displaying progress:

```markdown
## Applying fixes...

- [1/X] verify-<name1>: `path/to/file.ts` fixed
- [2/X] verify-<name2>: `path/to/file.tsx` fixed

X fixes completed.
```

**If "Fix individually" is selected:**

For each issue, show the fix content and use `AskUserQuestion` to confirm approval.

### Step 6: Post-Fix Re-verification

If fixes were applied, re-run only the skills that had issues and compare Before/After:

```markdown
## Post-Fix Re-verification

Re-running skills that had issues...

| Verification Skill | Before Fix | After Fix |
|--------------------|------------|-----------|
| verify-<name1> | X issues | PASS |
| verify-<name2> | Y issues | PASS |

All verifications passed!
```

**If issues remain:**

```markdown
### Remaining Issues

| # | Skill | File | Problem |
|---|-------|------|---------|
| 1 | verify-<name> | `path/to/file.ts:42` | Cannot auto-fix — manual review required |

Resolve manually and run `/verify-implementation` again.
```

---

## Exceptions

The following are **not issues**:

1. **Projects with no registered skills** — Display a guidance message and exit, not an error
2. **Skill-specific exceptions** — Patterns defined in each verify skill's Exceptions section are not reported as issues
3. **verify-implementation itself** — Do not include itself in the skills to execute list
4. **manage-skills** — Does not start with `verify-` so is not included in execution targets

## Related Files

| File | Purpose |
|------|---------|
| `.claude/skills/manage-skills/SKILL.md` | Skill maintenance (manages the skills to execute list in this file) |
| `CLAUDE.md` | Project guidelines |
