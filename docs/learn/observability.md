<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Observability: logs and monitoring

Observability is key to getting clearer, more verbose output during the tuning process. This page covers the logging level and the `AITUNE_CONSOLE_OUTPUT` environment variable (with hardware monitoring to come).

## Logging level: `logging.INFO`

AITune emits progress messages at **INFO** level (strategy steps, backend selection, build status, validation, etc.). If the root or `aitune` logger is left at its default level (often WARNING), those messages are not shown.

**Enable INFO for better verbosity:**

```python
import logging

logging.basicConfig(level=logging.INFO, force=True)
```

With INFO (or DEBUG for more detail), you will see messages such as:

- Which strategy and backend are running
- Build and validation progress
- Selected backend and high-level timing

## Environment variable: `AITUNE_CONSOLE_OUTPUT`

During tuning, backend builds (e.g. TensorRT) run inside an **output control** context. By default, their stdout, stderr, and logging are **not** printed to the console; they are redirected to a build log file (e.g. `build.log` in the backend cache directory). That keeps the terminal quiet and avoids flooding it with compiler/build logs.

To see that backend output **live on the console** as well, set:

```bash
export AITUNE_CONSOLE_OUTPUT=1
```

Accepted values (case-insensitive): `1`, `true`, `yes`. If unset or any other value, console output from the build phase remains disabled (output only in the log file).

**Behavior summary:**

| `AITUNE_CONSOLE_OUTPUT` | Console during backend build | Build log file        |
|-------------------------|-----------------------------|------------------------|
| Unset / 0 / false        | No backend output           | Yes (e.g. `build.log`) |
| 1 / true / yes           | Backend output shown        | Yes (tee: console + file) |

So:

- **Default (unset):** Cleaner terminal; inspect `build.log` under the backend cache dir if you need build details.
- **Set to 1:** More verbose terminal; same content is also written to the log file.

## Using both for maximum verbosity

For the most visible tuning process:

1. Set the logging level to INFO (or DEBUG) in your script so AITune’s own messages are shown.
2. Set `AITUNE_CONSOLE_OUTPUT=1` in the environment so backend build output (e.g. TensorRT) is shown on the console and still captured in the build log.

```bash
export AITUNE_CONSOLE_OUTPUT=1
python your_tuning_script.py
```

```python
import logging

logging.basicConfig(level=logging.INFO, force=True)

# ... your tuning code ...
```

See also `aitune.utils.setup_logging` / `control_output` in the API.

