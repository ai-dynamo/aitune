<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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

# Deployment Guide

This guide covers deploying AITune-tuned models in production environments, from saving tuned models to loading them in production systems.

## Overview

AITune provides comprehensive tools for model deployment:

- **Save/Load**: Persist and restore tuned models
- **Storage Options**: Local and custom storage backends
- **Verification**: SHA256 checksums for integrity
- **Portability**: Deploy across different environments

## Quick Start

### Save a Tuned Model

```python
import aitune.torch as ait

# After tuning
ait.save(tuned_model, "model.ait")
```

### Load a Tuned Model

```python
import aitune.torch as ait

# In production
model = YourModel()
model = ait.load(model, "model.ait")
output = model(input_data)
```

## Saving Tuned Models

### Basic Save

```python
import aitune.torch as ait

# Save after tuning
ait.save(model, "checkpoints/model.ait")
```

This creates:

- `checkpoints/model.ait`: Compressed checkpoint with tuned modules
- `checkpoints/model_sha256_sums.txt`: SHA256 checksums
- `checkpoints/model/`: Decompressed artifacts (after first load)

### With Custom Storage

```python
from aitune.torch import LocalTorchStorage

# Configure storage
storage = LocalTorchStorage(
    base_folder="production/models",
    remove_checkpoint_after_tune=False,  # Keep intermediate files
)

# Save with custom storage
ait.save(model, "model_v2.ait", storage=storage)
```

## Loading Tuned Models

### Basic Load

```python
import aitune.torch as ait

# Create model instance
model = YourModel()
model.eval()
model.to("cuda")

# Load tuned version
ait.load(model, "checkpoints/model.ait")

# Ready for inference
output = model(input_data)
```

### With Custom Storage

```python
from aitune.torch import LocalTorchStorage

storage = LocalTorchStorage(base_folder="production/models")
ait.load(model, "model.ait", storage=storage)
```

### Loading Process

1. First Load:

- Decompresses `.ait` file
- Extracts artifacts to `checkpoints/` directory
- Verifies checksums
- Loads backend and weights
- Slower (decompression overhead)

2. Subsequent Loads:

- Uses decompressed files from `checkpoints/`
- Skips decompression
- Faster startup

## Next Steps

- Review [AOT Tuning](../aot_tuning.md) for tuning best practices
- Explore [Tune Strategies](../tune_strategies/tune_strategies.md) for optimization
- Check [Backend Guides](../backends/tensorrt_backend.md) for backend-specific deployment notes
