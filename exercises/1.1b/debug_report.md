# Sub-exercise 4: Debug Report

**Module:** 1.1b | **Kernel:** Three Buggy Matmul Variants

Document your debugging process for each of the 3 bugs. Use the diagnostic flowchart
from `debugging-methodology.md`. Run: `python exercise_debug.py [1|2|3|all]`

**Note on FP32 verification:** `tl.dot` on Ampere silently uses tensor cores, which
downcast FP32 to FP16. FP32 verification is not viable for `tl.dot`-based kernels on
this hardware. Skip Step 2 of the flowchart and go directly to pattern analysis.

---

## Bug 1

### Pass/Fail Pattern

| Test Case | M | N | K | Result | Max Error |
|-----------|---|---|---|--------|-----------|
| TODO | | | | | |

**Pattern observed:** TODO — which cases pass? Which fail? What do the failing cases have in common?

### Tolerance Sweep (first failing case)

```
TODO: paste tolerance sweep output
```

**Interpretation:** TODO — gradual falloff (precision) or sharp cliff (bug)?

### Diagnosis

**Bug identified:** TODO
**Evidence:** TODO
**Fix:** TODO — what line would you change and how?

---

## Bug 2

### Pass/Fail Pattern

| Test Case | M | N | K | Result | Max Error |
|-----------|---|---|---|--------|-----------|
| TODO | | | | | |

**Pattern observed:** TODO — which cases pass? Which fail? What's special about the passing case?

### Tolerance Sweep (first failing case)

```
TODO: paste tolerance sweep output
```

**Interpretation:** TODO

### Diagnosis

**Bug identified:** TODO
**Evidence:** TODO
**Fix:** TODO — what line would you change and how?

---

## Bug 3

### Pass/Fail Pattern

| Test Case | M | N | K | Result | Max Error |
|-----------|---|---|---|--------|-----------|
| TODO | | | | | |

**Pattern observed:** TODO — how does this pattern compare to Bug 2?

### Tolerance Sweep (first failing case)

```
TODO: paste tolerance sweep output
```

**Interpretation:** TODO

### Diagnosis

**Bug identified:** TODO
**Evidence:** TODO
**Fix:** TODO — what line would you change and how?

---

## Summary

### Bug Classification

| Bug | Category | Diagnostic Signal |
|-----|----------|-------------------|
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |

### Lessons Learned

1. What debugging technique was most useful across all 3 bugs? TODO
2. How did the `K = BLOCK_SIZE_K` control case help? TODO
3. What would you look for first next time you see a failing matmul kernel? TODO
