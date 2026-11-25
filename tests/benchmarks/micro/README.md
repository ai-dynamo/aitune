# Microbenchmarks

Measure CPU tensor flattening, output reconstruction, and restoration overhead.
These measurements exclude model execution and do not measure end-to-end inference.

Run from the repository root:

```bash
python -m pytest -q -s -n 0 tests/benchmarks/micro \
    -o junit_family=legacy \
    --junitxml=microbenchmark-results.xml
```

Compare matching cases on the same idle machine and software environment, without
coverage. Results include repeated samples, median microseconds per call, and
environment details; no latency thresholds are enforced.

CI runs the **Microbenchmarks** job and uploads the `microbenchmark-results` artifact.
