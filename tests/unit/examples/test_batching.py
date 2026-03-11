# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from examples.common.aitune_examples_common.batching import batch


@pytest.mark.parametrize(
    "batch_size,expected_counter",
    [
        (1, 1 + 2 + 3),  # 3 batches of 1
        (2, (1 + 2) * 2 + 3 * 1),  # 2 batches of 2 and 1
        (4, (1 + 2 + 3) * 3),  # all in one batch
    ],
)
def test_batch_and_async_call(batch_size, expected_counter):
    import asyncio

    async def main():
        assert asyncio.get_running_loop() is not None

        class DummyClass:
            def __init__(self):
                self.counter = 0

            @batch(max_batch_size=batch_size, batch_wait_timeout_s=0.1)
            async def batched_generator(self, x: list):
                self.counter += sum(x) * len(x)
                return x

        obj = DummyClass()

        try:
            t1 = asyncio.create_task(obj.batched_generator(1))
            t2 = asyncio.create_task(obj.batched_generator(2))
            t3 = asyncio.create_task(obj.batched_generator(3))

            await t1
            await t2
            await t3

            assert obj.counter == expected_counter

        finally:
            obj.batched_generator.finish()

    asyncio.run(main())
