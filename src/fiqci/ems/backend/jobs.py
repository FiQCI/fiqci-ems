from __future__ import annotations

import logging
import threading
import time
from typing import Any, NamedTuple
from collections.abc import Callable

from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result

logger: logging.Logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED})


class BatchFailedError(RuntimeError):
	"""Raised when one or more batches of a submitted job fail.

	The message identifies the failing batch by its index and the range of original circuit
	indices it covered, so callers can map the failure back to their input.
	"""


class _UnsubmittedBatch:
	"""Stand-in for a batch that was not successfully submitted to the backend.

	Submission is not atomic: when a batch is rejected mid-stream, ``run()`` stops submitting and
	still returns a handle covering every intended batch. The rejected batch and any batches skipped
	after it are represented by this placeholder so the handle exposes a uniform, index-aligned
	per-batch view. It reports a terminal status (``ERROR`` for the batch that failed to submit,
	``CANCELLED`` for batches skipped afterwards), has no backend job id, and raises when its result
	is requested.
	"""

	def __init__(self, circuit_range: tuple[int, int], status: JobStatus, error: str | None = None) -> None:
		self._range = circuit_range
		self._status = status
		self._error = error

	def job_id(self) -> None:
		"""Unsubmitted batches have no backend job id."""
		return None

	def status(self) -> JobStatus:
		return self._status

	def result(self, timeout: float | None = None) -> Result:
		start, end = self._range
		detail = f": {self._error}" if self._error else ""
		raise BatchFailedError(f"Batch covering circuits {start}-{end - 1} was not submitted{detail}")


class PartialBatch(NamedTuple):
	"""Snapshot of a single batch within a multi-batch job.

	Attributes:
		index: Position of the batch in submission order.
		circuit_range: ``(start, end_exclusive)`` original circuit indices covered by the batch.
		status: Current :class:`~qiskit.providers.JobStatus` of the batch job.
		job_id: Backend job id of the batch.
		result: The batch's :class:`~qiskit.result.Result` if it has completed, else ``None``.
	"""

	index: int  # type: ignore[bad-override]  # NamedTuple field shadows the read-write tuple.index method
	circuit_range: tuple[int, int]
	status: JobStatus
	job_id: str
	result: Result | None
	
class BatchedJob:
	"""Lazy handle over one or more backend jobs submitted as ordered batches.

	A larger circuit list is split into batches that are each submitted to the backend; this
	wrapper holds the resulting per-batch jobs. It is returned immediately from
	:meth:`FiQCIBackend.run` (the submission loop does not wait for results), so callers can
	inspect ``job_ids()`` and poll ``status()``/``done()`` right away.

	Calling :meth:`result` blocks until every batch reaches a terminal state, then concatenates
	the batches' ``results`` lists in submission order (so ``get_counts(idx)`` on the combined
	Result corresponds to the original circuit index) and runs the optional ``post_process``
	callback that applies error mitigation. The combined/post-processed result is computed once
	and cached. If any batch failed, :meth:`result` raises :class:`BatchFailedError` identifying
	the batch and the original circuit indices it covered.
	"""

	def __init__(
		self,
		jobs: list[JobV1],
		batch_ranges: list[tuple[int, int]] | None = None,
		post_process: Callable[[Result], Result] | None = None,
	) -> None:
		"""Initialize the handle.

		Args:
			jobs: Per-batch jobs in submission order. Must be non-empty.
			batch_ranges: ``(start, end_exclusive)`` original circuit-index range for each batch,
				used for partial-result reporting and failure messages. If omitted, ranges are
				reported as best-effort placeholders.
			post_process: Optional callback mapping the combined raw Result to the final
				(mitigated) Result. Runs once on the first :meth:`result` call.
		"""
		assert jobs, "BatchedJob must wrap at least one job"
		self._jobs = jobs
		self._batch_ranges = batch_ranges
		self._post_process = post_process
		self._combined_result: Result | None = None
		self._final_result: Result | None = None
		self._lock = threading.Lock()

	# -- identity / polling (available immediately, before any results) --

	def job_id(self) -> str:
		"""Backend job id of the first batch (for single-job back-compat)."""
		return self._jobs[0].job_id()

	def job_ids(self) -> list[str]:
		"""Backend job ids of every batch, in submission order."""
		return [job.job_id() for job in self._jobs]

	def statuses(self) -> list[JobStatus]:
		"""Current :class:`JobStatus` of each batch, in submission order."""
		return [job.status() for job in self._jobs]

	def status(self) -> JobStatus:
		"""Single aggregated status across all batches (see :meth:`_aggregate_status`)."""
		return self._aggregate_status(self.statuses())

	def done(self) -> bool:
		"""True once every batch has reached a terminal state (DONE/ERROR/CANCELLED)."""
		return all(status in _TERMINAL_STATUSES for status in self.statuses())

	def all_succeeded(self) -> bool:
		"""True once every batch has completed successfully (DONE)."""
		return all(status == JobStatus.DONE for status in self.statuses())

	@staticmethod
	def _aggregate_status(statuses: list[JobStatus]) -> JobStatus:
		"""Collapse per-batch statuses into one by priority.

		A failure anywhere dominates (ERROR, then CANCELLED). Otherwise the job is DONE only when
		all batches are done; if any batch is still progressing the least-advanced active state is
		reported (RUNNING > VALIDATING > QUEUED > INITIALIZING).
		"""
		if not statuses:
			return JobStatus.DONE
		if any(status == JobStatus.ERROR for status in statuses):
			return JobStatus.ERROR
		if any(status == JobStatus.CANCELLED for status in statuses):
			return JobStatus.CANCELLED
		if all(status == JobStatus.DONE for status in statuses):
			return JobStatus.DONE
		for active in (JobStatus.RUNNING, JobStatus.VALIDATING, JobStatus.QUEUED):
			if any(status == active for status in statuses):
				return active
		return JobStatus.INITIALIZING

	def _range(self, index: int) -> tuple[int, int]:
		"""Original circuit-index range for batch ``index`` (best-effort if unknown)."""
		if self._batch_ranges is not None and index < len(self._batch_ranges):
			return self._batch_ranges[index]
		return (index, index + 1)

	# -- partial results (batch-granular) --

	def partial_results(self) -> list[PartialBatch]:
		"""Per-batch snapshot, exposing results for batches that have already completed.

		Each entry carries the batch's status and, for completed (DONE) batches, its Result.
		Batches that are still running or have failed report ``result=None``. Results are exposed
		at batch granularity only; the globally-indexed combined Result is available from
		:meth:`result` once all batches are terminal.
		"""
		snapshots: list[PartialBatch] = []
		for index, job in enumerate(self._jobs):
			status = job.status()
			result: Result | None = None
			if status == JobStatus.DONE:
				try:
					result = job.result()
				except Exception:  # pragma: no cover - defensive
					result = None
			snapshots.append(
				PartialBatch(
					index=index, circuit_range=self._range(index), status=status, job_id=job.job_id(), result=result
				)
			)
		return snapshots

	# -- combined / post-processed result --

	def result(self, timeout: float | None = None) -> Result:
		"""Return the combined, post-processed result, computed once and cached.

		Blocks until every batch is terminal. Raises :class:`BatchFailedError` if any batch
		failed, otherwise concatenates batch results in submission order and applies the
		``post_process`` callback (if any).

		Args:
			timeout: Best-effort total budget (seconds) shared across all batches.
		"""
		if self._final_result is not None:
			return self._final_result
		with self._lock:
			if self._final_result is not None:
				return self._final_result
			combined = self._combine(timeout)
			self._final_result = self._post_process(combined) if self._post_process is not None else combined
			return self._final_result

	def _combine(self, timeout: float | None) -> Result:
		"""Wait for all batches and concatenate their results in submission order."""
		if self._combined_result is not None:
			return self._combined_result

		from qiskit.result import Result as QiskitResult

		deadline = None if timeout is None else time.monotonic() + timeout
		results: list[Result] = []
		failures: list[tuple[int, str, str]] = []
		for index, job in enumerate(self._jobs):
			remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
			try:
				# Concrete IQM jobs accept a timeout; the abstract JobV1.result stub does not declare one.
				result = job.result() if remaining is None else job.result(remaining)  # type: ignore[bad-argument-count]
			except Exception as exc:  # batch failed at the backend
				failures.append((index, job.job_id(), str(exc)))
				continue
			if job.status() in (JobStatus.ERROR, JobStatus.CANCELLED):
				failures.append((index, job.job_id(), str(job.status())))
				continue
			results.append(result)

		if failures:
			raise BatchFailedError(self._failure_message(failures))

		combined_data: dict[str, Any] = results[0].to_dict()
		merged_results: list[Any] = list(combined_data.get("results") or [])
		for result in results[1:]:
			merged_results.extend(result.to_dict().get("results") or [])
		combined_data["results"] = merged_results
		self._combined_result = QiskitResult.from_dict(combined_data)
		return self._combined_result

	def _failure_message(self, failures: list[tuple[int, str, str]]) -> str:
		"""Build a human-readable message naming each failed batch and its circuit range."""
		parts = []
		for index, job_id, detail in failures:
			start, end = self._range(index)
			parts.append(f"batch {index} (circuits {start}-{end - 1}, job_id={job_id}): {detail}")
		return f"{len(failures)} of {len(self._jobs)} batch(es) failed: " + "; ".join(parts)

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access to the first underlying job."""
		return getattr(self._jobs[0], name)


class MitigatedJob:
	"""Lazy view over a :class:`BatchedJob` whose result has error mitigation applied.

	The mitigation work is deferred to the wrapped handle's ``post_process`` callback, so this
	wrapper simply exposes the handle's polling API and a :meth:`result` that returns the
	mitigated, combined Result. It exists as a distinct type so callers and tests can detect that
	mitigation was configured for the run.
	"""

	def __init__(self, handle: BatchedJob) -> None:
		"""Initialize the wrapper.

		Args:
			handle: The submission handle carrying the batch jobs and the mitigation
				``post_process`` callback.
		"""
		self._handle = handle

	def result(self, timeout: float | None = None) -> Result:
		"""Return the mitigated, combined result (computed once by the underlying handle)."""
		return self._handle.result(timeout)

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access (status, done, job_ids, partial_results, …) to the handle."""
		return getattr(self._handle, name)
