# Results Summary

## Evaluation design

The retained development evidence contains 871 numerical FinQA examples evaluated under 13 methods, giving 11,323 method/example records.

## Main system-level comparison

| Measure | Unconstrained baseline | Selector-guided ProofLock |
|---|---:|---:|
| Correct numerical answers | 6/871 | 14/871 |
| Numerical accuracy | 0.69% | 1.61% |
| Complete numerical outputs rejected by certificate policy | 46 | 0 |
| Candidate recall | — | 424/871 (48.68%) |
| Correct selection conditional on recall | — | 14/424 (3.30%) |

The 14-versus-6 comparison is a **system-level** comparison because selector-guided ProofLock changes candidate construction, ranking and decoding together. It should not be interpreted as the isolated causal effect of the trie.

## Structural enforcement

The fragment-token ablation produced 92 certificate-policy escapes, whereas the evaluated whole-answer hard locks produced none in the retained development evidence. This supports the engineering conclusion that complete-answer locking correctly instantiated the encoded admissibility policy under the observed execution.

The lock and verifier share the same policy; therefore zero policy-invalid hard-lock outputs are primarily an implementation invariant rather than independent proof of exhaustive documentary provenance.

## Semantic bottlenecks

1. **Context/evidence access:** 870/871 candidate-rich prompts exceeded the model's 512-token input budget.
2. **Candidate sufficiency:** the reference answer was present in the final candidate lattice for 424/871 examples.
3. **Semantic selection:** only 14 of those 424 reachable references were selected correctly.
4. **Output enforcement:** whole-answer locks prevented policy-invalid complete outputs but could not determine whether a legal candidate was semantically appropriate.

## Release decision

The provenance-policy release gate was feasible under the implemented metric, but no semantic-risk threshold met the predefined release requirements. At the best eligible semantic threshold, 186 answers would have been accepted and 178 were wrong. The public test therefore remained untouched.

## Interpretation

The project should be read as a provenance-control and failure-analysis study rather than a claim of competitive financial-QA performance.
