# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark all scenarios"""

import logging
from logging import basicConfig
from time import perf_counter

import pandas as pd

from llm.benchmark import BenchmarkArgs, benchmark_scenario
from llm.cmd_args import get_benchmark_all_parser


def main():
    basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    parsed_args = get_benchmark_all_parser().parse_args()

    # baseline scenario: no cache, vanilla scenario
    args = BenchmarkArgs(
        model_id=parsed_args.model_id,
        sequence_lengths=parsed_args.sequence_lengths,
        iterations=1,
        warmup_iters=1,
        cache="no_cache",
        scenario="vanilla",
    )

    results = []
    start = perf_counter()

    if parsed_args.run_baseline:
        logging.info("Benchmarking baseline scenario: %s", args)
        results.extend(benchmark_scenario(args))

    for scenario in ["aot", "vanilla"]:
        for cache in ["static"]:
            args.cache = cache
            args.scenario = scenario
            logging.info("Benchmarking scenario: %s", args)
            results.extend(benchmark_scenario(args))

    df = pd.DataFrame(results)
    df.to_csv("benchmark_all.csv", index=False)
    df.drop(columns=["disable_compile"], inplace=True)
    duration = perf_counter() - start
    logging.info("Benchmarking took: %.1f seconds, results:\n%s", duration, df)


if __name__ == "__main__":
    main()
