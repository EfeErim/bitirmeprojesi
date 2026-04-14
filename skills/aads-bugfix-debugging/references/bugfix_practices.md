# Bugfix Practices

Load this reference only when you need literature-backed rationale for how to investigate or harden a bug fix.

## Repo-first rule

- Repo code, tests, and maintained docs define current AADS behavior.
- The papers and standards below justify debugging and hardening techniques. They do not override repo contracts.

## Recommended patterns

### 1. Reproduce and minimize before patching

- Use the smallest deterministic failing input, fixture, or diff you can produce.
- Delta debugging is the literature-backed pattern for isolating the minimal failure-inducing change or input.
- Source: Andreas Zeller, "Yesterday, my program worked. Today, it does not. Why?" ESEC/FSE 1999. DOI: `10.1145/318773.318946`
- Source: Andreas Zeller and Ralf Hildebrandt, "Simplifying and Isolating Failure-Inducing Input," IEEE TSE 2002. DOI: `10.1109/32.988498`

### 2. Turn the bug into an executable regression check

- Prefer a narrow failing test, schema validation, or contract check over a prose-only bug note.
- In this repo, the first choice is usually a targeted `pytest` case close to the touched module or workflow facade.
- This is an engineering practice reinforced by the techniques below: property checks, metamorphic checks, assertions, and mutation testing all work best once the failure is executable.

### 3. Use property-based testing for edge cases and invariants

- Property-based testing is useful when a function must preserve invariants across many inputs and edge cases, not just a few hand-picked examples.
- For Python, Hypothesis is the practical implementation; it also shrinks failures toward smaller counterexamples.
- Source: Koen Claessen and John Hughes, "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs," ICFP 2000.
- Source: David R. MacIver, Zac Hatfield-Dodds, and contributors, "Hypothesis: A new approach to property-based testing," Journal of Open Source Software 2019. DOI: `10.21105/joss.01891`

### 4. Use metamorphic testing for silent or oracle-poor bugs

- If you cannot write an exact oracle, test relations that should stay true across transformed inputs: round-trips, monotonicity, permutation invariance, stable ordering, or equivalent status transitions.
- This is especially useful for silent failures where the code returns plausible-looking but wrong outputs.
- Source: Tsong Yueh Chen et al., "Metamorphic Testing for Cybersecurity," IEEE Computer 2016. DOI: `10.1109/MC.2016.176`
- Source: Sergio Segura et al., "A Survey on Metamorphic Testing," IEEE Transactions on Software Engineering 2016. DOI: `10.1109/TSE.2016.2532875`

### 5. Check whether the test suite can actually kill faults

- A passing regression test is necessary but not always sufficient. When coverage looks good yet bugs still escape, use mutation-style thinking on the touched logic.
- Keep it targeted. This repo does not need blanket mutation tooling for every fix.
- Source: Mike Papadakis et al., "Mutation Testing Advances: An Analysis and Survey," Advances in Computers 2019. DOI: `10.1016/bs.adcom.2018.03.015`

### 6. Add assertions and type-backed boundaries to fail fast

- Assertions help surface corrupted state near the fault instead of letting it propagate into a later, harder-to-debug failure.
- Type hints support stronger offline analysis and refactoring on boundary-heavy code.
- Source: Gunnar Kudrjavets, Nachiappan Nagappan, and Thomas Ball, "Assessing the Relationship between Software Assertions and Code Quality: An Empirical Investigation," Microsoft Research Technical Report MSR-TR-2006-54.
- Source: PEP 484, "Type Hints," Python Standards Track. `https://peps.python.org/pep-0484/`
- Source: PEP 482, "Literature Overview for Type Hints," Python Informational PEP. `https://peps.python.org/pep-0482/`

## How to map this into AADS

- Training and readiness bugs: strengthen config normalization, artifact writers, and readiness invariants before adding broad fallbacks.
- Inference bugs: protect payload contracts, router status transitions, adapter lookup, and OOD attachment rules with explicit guards and tests.
- Notebook bugs: keep wrappers thin and validate the underlying script or workflow contract instead of papering over notebook state.
