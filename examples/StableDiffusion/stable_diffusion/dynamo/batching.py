# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023 Ray Authors
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
"""Dynamic batching decorator for async endpoints in AI Dynamo."""

import asyncio
import inspect
import io
import logging
import os
import sys
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache, wraps
from inspect import Parameter, isasyncgenfunction, iscoroutinefunction
from typing import Any

logger = logging.getLogger(__name__)

MAX_ONGOING_REQUESTS = 8

# The user can return these values in their streaming batch handler function to
# indicate that a request is finished, so Serve can terminate the request.
USER_CODE_STREAMING_SENTINELS = [StopIteration, StopAsyncIteration]


@dataclass
class _SingleRequest:
    self_arg: Any
    flattened_args: list[Any]
    future: asyncio.Future


@dataclass
class _GeneratorResult:
    result: Any
    next_future: asyncio.Future


def batch(  # noqa: C901
    _func: Callable | None = None,
    /,
    max_batch_size: int = 10,
    batch_wait_timeout_s: float = 0.0,
) -> Callable:
    """Converts a function to asynchronously handle batches.

    The function can be a standalone function or a class method. In both
    cases, the function must be `async def` and take a list of objects as
    its sole argument and return a list of the same length as a result.

    When invoked, the caller passes a single object. These will be batched
    and executed asynchronously once there is a batch of `max_batch_size`
    or `batch_wait_timeout_s` has elapsed, whichever occurs first.

    Example:

    .. code-block:: python
            from stable_diffusion.dynamo.batching import batch

            class BatchedDeployment:

                @batch(max_batch_size=2, batch_wait_timeout_s=0.1)
                async def batch_handler(self, requests: list[Request]) -> list[str]:
                    response_batch = []
                    for r in requests:
                        name = (await r.json())["name"]
                        response_batch.append(f"Hello {name}!")

                    return response_batch

                @endpoint()
                async def generate_image(self, req: Request) -> str:
                    return await self.batch_handler(req)

    Arguments:
        max_batch_size: the maximum batch size that will be executed in
            one call to the underlying function.
        batch_wait_timeout_s: the maximum duration to wait for
            `max_batch_size` elements before running the current batch.
    """
    # `_func` will be None in the case when the decorator is parametrized.
    # See the comment at the end of this function for a detailed explanation.
    if _func is not None:
        if not callable(_func):
            raise TypeError("@batch can only be used to decorate functions or methods.")

        if not iscoroutinefunction(_func):
            raise TypeError("Functions decorated with @batch must be 'async def'")

    _validate_max_batch_size(max_batch_size)
    _validate_batch_wait_timeout_s(batch_wait_timeout_s)

    def _batch_decorator(_func):
        lazy_batch_queue_wrapper = _LazyBatchQueueWrapper(
            max_batch_size,
            batch_wait_timeout_s,
            _func,
        )

        async def batch_handler_generator(
            first_future: asyncio.Future,
        ) -> AsyncGenerator:
            """Generator that handles generator batch functions."""
            future = first_future
            while True:
                try:
                    async_response: _GeneratorResult = await future
                    future = async_response.next_future
                    yield async_response.result
                except StopAsyncIteration:
                    break

        def enqueue_request(args, kwargs) -> asyncio.Future:
            signature_parameters = list(inspect.signature(_func).parameters.values())
            flattened_args: list = _flatten_args(signature_parameters, args, kwargs)

            # If the function is a method, remove self as an argument.
            self = _extract_self_if_method_call(args, _func)
            if self is not None:
                flattened_args = flattened_args[2:]

            batch_queue = lazy_batch_queue_wrapper.queue

            future = get_or_create_event_loop().create_future()
            batch_queue.put(_SingleRequest(self, flattened_args, future))
            return future

        @wraps(_func)
        def generator_batch_wrapper(*args, **kwargs):
            first_future = enqueue_request(args, kwargs)
            return batch_handler_generator(first_future)

        @wraps(_func)
        async def batch_wrapper(*args, **kwargs):
            # This will raise if the underlying call raised an exception.
            return await enqueue_request(args, kwargs)

        if isasyncgenfunction(_func):
            wrapper = generator_batch_wrapper
        else:
            wrapper = batch_wrapper

        # Store debugging methods in the lazy_batch_queue wrapper
        wrapper.get_curr_iteration_start_time = lazy_batch_queue_wrapper.get_curr_iteration_start_time
        wrapper.is_batching_task_alive = lazy_batch_queue_wrapper.is_batching_task_alive
        wrapper.finish = lazy_batch_queue_wrapper.finish

        return wrapper

    # Unfortunately, this is required to handle both non-parametrized
    # (@batch) and parametrized (@batch(**kwargs)) usage.
    # In the former case, `batch` will be called with the underlying
    # function as the sole argument. In the latter case, it will first be
    # called with **kwargs, then the result of that call will be called
    # with the underlying function as the sole argument (i.e., it must be a
    # "decorator factory.").
    return _batch_decorator(_func) if callable(_func) else _batch_decorator


class _BatchQueue:
    def __init__(
        self,
        max_batch_size: int,
        batch_wait_timeout_s: float,
        handle_batch_func: Callable | None = None,
    ) -> None:
        """Async queue that accepts individual items and returns batches.

        Respects max_batch_size and timeout_s; a batch will be returned when
        max_batch_size elements are available or the timeout has passed since
        the previous get.

        If handle_batch_func is passed in, a background coroutine will run to
        poll from the queue and call handle_batch_func on the results.

        Cannot be pickled.

        Arguments:
            max_batch_size: max number of elements to return in a batch.
            timeout_s: time to wait before returning an incomplete
                batch.
            batch_wait_timeout_s: time to wait before returning an incomplete
                batch.
            handle_batch_func(Optional[Callable]): callback to run in the
                background to handle batches if provided.
        """
        self.queue: asyncio.Queue[_SingleRequest] = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.requests_available_event = asyncio.Event()

        # Used for observability.
        self.curr_iteration_start_time = time.monotonic()

        self._handle_batch_task = None
        self._loop = get_or_create_event_loop()
        if handle_batch_func is not None:
            self._handle_batch_task = self._loop.create_task(self._process_batches(handle_batch_func))
        self._warn_if_max_batch_size_exceeds_max_ongoing_requests()

    def _warn_if_max_batch_size_exceeds_max_ongoing_requests(self):
        """Helper to check whether the max_batch_size is bounded.

        Log a warning to configure `max_ongoing_requests` if it's bounded.
        """
        max_ongoing_requests = get_max_ongoing_requests()
        if max_ongoing_requests < self.max_batch_size:
            logger.warning(
                "`max_batch_size` (%d) is larger than "
                "`max_ongoing_requests` (%d). This means "
                "the replica will never receive a full batch. Please update "
                "`max_ongoing_requests` to be >= `max_batch_size`.",
                self.max_batch_size,
                max_ongoing_requests,
            )

    def put(self, request: _SingleRequest) -> None:
        self.queue.put_nowait(request)
        self.requests_available_event.set()

    async def wait_for_batch(self) -> list[Any]:
        """Wait for batch respecting self.max_batch_size and self.timeout_s.

        Returns a batch of up to self.max_batch_size items. Waits for up to
        to self.timeout_s after receiving the first request that will be in
        the next batch. After the timeout, returns as many items as are ready.

        Always returns a batch with at least one item - will block
        indefinitely until an item comes in.
        """
        batch = []
        batch.append(await self.queue.get())

        # Cache current max_batch_size and batch_wait_timeout_s for this batch.
        max_batch_size = self.max_batch_size
        batch_wait_timeout_s = self.batch_wait_timeout_s

        # Wait self.timeout_s seconds for new queue arrivals.
        batch_start_time = time.monotonic()
        while True:
            remaining_batch_time_s = max(batch_wait_timeout_s - (time.monotonic() - batch_start_time), 0)
            try:
                # Wait for new arrivals.
                await asyncio.wait_for(self.requests_available_event.wait(), remaining_batch_time_s)
            except asyncio.TimeoutError:
                pass

            # Add all new arrivals to the batch.
            while len(batch) < max_batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())

            # Only clear the put event if the queue is empty. If it's not empty
            # we can start constructing a new batch immediately in the next loop.
            # The code that puts items into the queue runs on the same event loop
            # as this code, so there's no race condition between the time we
            # get objects in the queue (and clear the event) and when objects
            # get added to the queue.
            if self.queue.empty():
                self.requests_available_event.clear()

            if time.monotonic() - batch_start_time >= batch_wait_timeout_s or len(batch) >= max_batch_size:
                break

        return batch

    def _validate_results(self, results: Iterable[Any], input_batch_length: int) -> None:
        if len(results) != input_batch_length:  # type: ignore
            raise RuntimeError(
                "Batched function doesn't preserve batch size. "
                f"The input list has length {input_batch_length} but the "
                f"returned list has length {len(results)}."  # type: ignore
            )

    async def _consume_func_generator(
        self,
        func_generator: AsyncGenerator,
        initial_futures: list[asyncio.Future],
        input_batch_length: int,
    ) -> None:
        """Consumes batch function generator.

        This function only runs if the function decorated with @batch
        is a generator.
        """
        FINISHED_TOKEN = None  # noqa: N806

        futures: deque[asyncio.Future | None] = deque(initial_futures)
        assert len(futures) == input_batch_length
        try:
            async for results in func_generator:
                self._validate_results(results, input_batch_length)
                for idx in range(input_batch_length):
                    result, future = results[idx], futures[0]

                    if future is FINISHED_TOKEN:
                        # This caller has already terminated.
                        futures.append(FINISHED_TOKEN)  # type: ignore
                    elif result in USER_CODE_STREAMING_SENTINELS:
                        # User's code returned sentinel. No values left
                        # for caller. Terminate iteration for caller.
                        _set_exception_if_not_done(future, StopAsyncIteration)
                        futures.append(FINISHED_TOKEN)
                    else:
                        next_future = get_or_create_event_loop().create_future()
                        _set_result_if_not_done(future, _GeneratorResult(result, next_future))
                        futures.append(next_future)

                    # Remove processed future. We remove the future at the very
                    # end of the loop to ensure that if an exception occurs,
                    # all pending futures will get set in the `except` block.
                    futures.popleft()

            for future in futures:
                if future is not FINISHED_TOKEN:
                    _set_exception_if_not_done(future, StopAsyncIteration)
        except Exception as e:
            for future in futures:
                if future is not FINISHED_TOKEN:
                    _set_exception_if_not_done(future, e)

    async def _assign_func_results(
        self,
        func_future: asyncio.Future,
        futures: list[asyncio.Future],
        input_batch_length: int,
    ):
        """Assigns func's results to the list of futures."""
        try:
            results = await func_future
            self._validate_results(results, input_batch_length)
            for result, future in zip(results, futures, strict=True):
                _set_result_if_not_done(future, result)
        except Exception as e:
            for future in futures:
                _set_exception_if_not_done(future, e)

    async def _process_batches(self, func: Callable) -> None:
        """Loops infinitely and processes queued request batches."""
        while not self._loop.is_closed():
            try:
                self.curr_iteration_start_time = time.monotonic()
                await self._process_batch(func)
            except Exception:
                logger.exception("_process_batches asyncio task ran into an unexpected exception.")

    async def _process_batch(self, func: Callable) -> None:
        """Processes queued request batch."""
        batch: list[_SingleRequest] = await self.wait_for_batch()
        # Remove requests that have been cancelled from the batch. If
        # all requests have been cancelled, simply return and wait for
        # the next batch.
        batch = [req for req in batch if not req.future.cancelled()]
        if len(batch) == 0:
            return

        futures = [item.future for item in batch]

        # Most of the logic in the function should be wrapped in this try-
        # except block, so the futures' exceptions can be set if an exception
        # occurs. Otherwise, the futures' requests may hang indefinitely.
        try:
            self_arg = batch[0].self_arg
            args, kwargs = _batch_args_kwargs([item.flattened_args for item in batch])

            # Method call.
            if self_arg is not None:
                func_future_or_generator = func(self_arg, *args, **kwargs)
            # Normal function call.
            else:
                func_future_or_generator = func(*args, **kwargs)

            if isasyncgenfunction(func):
                func_generator = func_future_or_generator
                await self._consume_func_generator(func_generator, futures, len(batch))
            else:
                func_future = func_future_or_generator
                await self._assign_func_results(func_future, futures, len(batch))

        except Exception as e:
            logger.exception("_process_batch ran into an unexpected exception.")

            for future in futures:
                _set_exception_if_not_done(future, e)

    def __del__(self):
        self.finish()

    def finish(self):
        if self._handle_batch_task is not None and get_or_create_event_loop().is_running():
            self._handle_batch_task.cancel()
        self._handle_batch_task = None


class _LazyBatchQueueWrapper:
    """Stores a _BatchQueue and updates its settings.

    _BatchQueue cannot be pickled, you must construct it lazily
    at runtime inside a replica. This class initializes a queue only upon
    first access.
    """

    def __init__(
        self,
        max_batch_size: int = 10,
        batch_wait_timeout_s: float = 0.0,
        handle_batch_func: Callable | None = None,
    ):
        self._queue: _BatchQueue | None = None
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.handle_batch_func = handle_batch_func

    @property
    def queue(self) -> _BatchQueue:
        """Returns _BatchQueue.

        Initializes queue when called for the first time.
        """
        if self._queue is None:
            self._queue = _BatchQueue(
                self.max_batch_size,
                self.batch_wait_timeout_s,
                self.handle_batch_func,
            )
        return self._queue

    def get_curr_iteration_start_time(self) -> float | None:
        """Gets current iteration's start time on default _BatchQueue implementation.

        Returns None if the batch handler doesn't use a default _BatchQueue.
        """
        if hasattr(self.queue, "curr_iteration_start_time"):
            return self.queue.curr_iteration_start_time
        else:
            return None

    async def is_batching_task_alive(self) -> bool:
        """Gets whether default _BatchQueue's background task is alive.

        Returns False if the batch handler doesn't use a default _BatchQueue.
        """
        if hasattr(self.queue, "_handle_batch_task"):
            return not self.queue._handle_batch_task.done()  # type: ignore
        else:
            return False

    async def get_handling_task_stack(self) -> str | None:
        """Gets the stack for the default _BatchQueue's background task.

        Returns empty string if the batch handler doesn't use a default _BatchQueue.
        """
        if hasattr(self.queue, "_handle_batch_task"):
            str_buffer = io.StringIO()
            self.queue._handle_batch_task.print_stack(file=str_buffer)  # type: ignore
            return str_buffer.getvalue()
        else:
            return None

    def finish(self):
        if self._queue is not None:
            self._queue.finish()
        self._queue = None


def _validate_max_batch_size(max_batch_size):
    if not isinstance(max_batch_size, int):
        if isinstance(max_batch_size, float) and max_batch_size.is_integer():
            max_batch_size = int(max_batch_size)
        else:
            raise TypeError(f"max_batch_size must be integer >= 1, got {max_batch_size}")

    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be an integer >= 1, got {max_batch_size}")


def _validate_batch_wait_timeout_s(batch_wait_timeout_s):
    if not isinstance(batch_wait_timeout_s, (float, int)):
        raise TypeError(f"batch_wait_timeout_s must be a float >= 0, got {batch_wait_timeout_s}")

    if batch_wait_timeout_s < 0:
        raise ValueError(f"batch_wait_timeout_s must be a float >= 0, got {batch_wait_timeout_s}")


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get a running async event loop if one exists, otherwise create one.

    This function serves as a proxy for the deprecating get_event_loop().
    It tries to get the running loop first, and if no running loop
    could be retrieved:
    - For python version <3.10: it falls back to the get_event_loop
        call.
    - For python version >= 3.10: it uses the same python implementation
        of _get_event_loop() at asyncio/events.py.

    Ideally, one should use high level APIs like asyncio.run() with python
    version >= 3.7, if not possible, one should create and manage the event
    loops explicitly.
    """
    vers_info = sys.version_info
    if vers_info.major >= 3 and vers_info.minor >= 10:
        # This follows the implementation of the deprecating `get_event_loop`
        # in python3.10's asyncio. See python3.10/asyncio/events.py
        # _get_event_loop()
        try:
            loop = asyncio.get_running_loop()
            assert loop is not None
            return loop
        except RuntimeError as e:
            # No running loop, relying on the error message as for now to
            # differentiate runtime errors.
            assert "no running event loop" in str(e)
            return asyncio.get_event_loop_policy().get_event_loop()

    return asyncio.get_event_loop()


def _set_result_if_not_done(future: asyncio.Future, result: Any):
    """Sets the future's result if the future is not done."""
    if not future.done():
        future.set_result(result)


def _set_exception_if_not_done(future: asyncio.Future, exception: Any):
    """Sets the future's exception if the future is not done."""
    if not future.done():
        future.set_exception(exception)


# This dummy type is also defined in ArgumentsBuilder.java. Please keep it
# synced.
DUMMY_TYPE = b"__DUMMY__"


def _validate_args(signature_parameters: list[Parameter], args, kwargs):
    """Validates the arguments against the signature.

    Args:
        signature_parameters: The list of Parameter objects
            representing the function signature, obtained from
            `extract_signature`.
        args: The positional arguments passed into the function.
        kwargs: The keyword arguments passed into the function.

    Raises:
        TypeError: Raised if arguments do not fit in the function signature.
    """
    reconstructed_signature = inspect.Signature(parameters=signature_parameters)
    try:
        reconstructed_signature.bind(*args, **kwargs)
    except TypeError as exc:  # capture a friendlier stacktrace
        raise TypeError(str(exc)) from None


def _flatten_args(signature_parameters: list[Parameter], args, kwargs):
    """Validates the arguments against the signature and flattens them.

    The flat list representation is a serializable format for arguments.
    Since the flatbuffer representation of function arguments is a list, we
    combine both keyword arguments and positional arguments. We represent
    this with two entries per argument value - [DUMMY_TYPE, x] for positional
    arguments and [KEY, VALUE] for keyword arguments. See the below example.
    See `recover_args` for logic restoring the flat list back to args/kwargs.

    Args:
        signature_parameters: The list of Parameter objects
            representing the function signature, obtained from
            `extract_signature`.
        args: The positional arguments passed into the function.
        kwargs: The keyword arguments passed into the function.

    Returns:
        List of args and kwargs. Non-keyword arguments are prefixed
            by internal enum DUMMY_TYPE.

    Raises:
        TypeError: Raised if arguments do not fit in the function signature.
    """
    _validate_args(signature_parameters, args, kwargs)
    list_args = []
    for arg in args:
        list_args += [DUMMY_TYPE, arg]

    for keyword, arg in kwargs.items():
        list_args += [keyword, arg]
    return list_args


def _recover_args(flattened_args):
    """Recreates `args` and `kwargs` from the flattened arg list.

    Args:
        flattened_args: List of args and kwargs. This should be the output of
            `flatten_args`.

    Returns:
        args: The non-keyword arguments passed into the function.
        kwargs: The keyword arguments passed into the function.
    """
    assert len(flattened_args) % 2 == 0, "Flattened arguments need to be even-numbered. See `flatten_args`."
    args = []
    kwargs = {}
    for name_index in range(0, len(flattened_args), 2):
        name, arg = flattened_args[name_index], flattened_args[name_index + 1]
        if name == DUMMY_TYPE:
            args.append(arg)
        else:
            kwargs[name] = arg

    return args, kwargs


def _batch_args_kwargs(list_of_flattened_args: list[list]) -> tuple[tuple, dict]:
    """Batch a list of flatten args and returns regular args and kwargs."""
    # Ray's flatten arg format is a list with alternating key and values
    # e.g. args=(1, 2), kwargs={"key": "val"} got turned into
    #      [None, 1, None, 2, "key", "val"]
    arg_lengths = {len(args) for args in list_of_flattened_args}
    assert len(arg_lengths) == 1, "All batch requests should have the same number of parameters."
    arg_length = arg_lengths.pop()

    batched_flattened_args = []
    for idx in range(arg_length):
        if idx % 2 == 0:
            batched_flattened_args.append(list_of_flattened_args[0][idx])
        else:
            batched_flattened_args.append([item[idx] for item in list_of_flattened_args])

    return _recover_args(batched_flattened_args)  # type: ignore


def _extract_self_if_method_call(args: Sequence, func: Callable) -> object | None:
    """Check if this is a method rather than a function.

    Does this by checking to see if `func` is the attribute of the first
    (`self`) argument under `func.__name__`. Unfortunately, this is the most
    robust solution to this I was able to find. It would also be preferable
    to do this check when the decorator runs, rather than when the method is.

    Returns the `self` object if it's a method call, else None.

    Arguments:
        args: arguments to the function/method call.
        func: the unbound function that was called.
    """
    if len(args) > 0:
        method = getattr(args[0], func.__name__, False)
        if method:
            wrapped = getattr(method, "__wrapped__", False)
            if wrapped and wrapped == func:
                return args[0]

    return None


@lru_cache
def get_max_ongoing_requests() -> int:
    """Get the max ongoing requests from the environment variable.

    Returns:
        The max ongoing requests.
    """
    return int(os.getenv("DYNAMIC_BATCHING_MAX_ONGOING_REQUESTS", MAX_ONGOING_REQUESTS))
