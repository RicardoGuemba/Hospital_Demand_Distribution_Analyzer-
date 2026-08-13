"""Application controller."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox, QWidget

from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.app.workers import (
    BulkFitWorker,
    ClusterWorker,
    FitWorker,
    ImportWorker,
    ReplenishmentQminWorker,
)
from hospital_demand_analyzer.clustering.cutoffs import DEFAULT_CUTOFF_FIELD, suggest_threshold_cutoffs
from hospital_demand_analyzer.clustering.features import build_feature_matrix
from hospital_demand_analyzer.clustering.threshold_clustering import run_threshold_clustering
from hospital_demand_analyzer.domain.demand_values import prepare_demand_values, sanitize_demand
from hospital_demand_analyzer.domain.models import ClusterResult
from hospital_demand_analyzer.outliers.engine import apply_outlier_method, apply_treatment
from hospital_demand_analyzer.services.cache import ParquetCache
from hospital_demand_analyzer.services.time_series import global_period_bounds
from hospital_demand_analyzer.statistics.fitter import fit_all
from hospital_demand_analyzer.statistics.fit_params import best_fit


class AppController:
    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        self.state = state
        self.parent = parent
        self.cache = ParquetCache(state.config.cache_path())
        self._import_worker: ImportWorker | None = None
        self._cluster_worker: ClusterWorker | None = None
        self._fit_workers: dict[str, FitWorker] = {}
        self._bulk_fit_worker: BulkFitWorker | None = None
        self._bulk_fit_cancel = False
        self._replenishment_qmin_worker: ReplenishmentQminWorker | None = None
        self._replenishment_qmin_cancel = False
        self._import_in_progress = False
        state.value_mode_changed.connect(self._on_value_mode_changed)

    def _on_value_mode_changed(self, _mode: str) -> None:
        self.state.fit_results.clear()
        for worker in list(self._fit_workers.values()):
            if worker.isRunning():
                worker.requestInterruption()
        self._fit_workers.clear()

    @property
    def import_in_progress(self) -> bool:
        return self._import_in_progress

    def _stop_worker(self, worker: QThread | None, timeout_ms: int = 120_000) -> None:
        if worker is None:
            return
        if worker.isRunning():
            worker.requestInterruption()
            worker.quit()
            worker.wait(timeout_ms)

    def _start_worker(self, worker: QThread, previous: QThread | None) -> None:
        self._stop_worker(previous)
        if self.parent is not None:
            worker.setParent(self.parent)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def shutdown(self, timeout_ms: int = 120_000) -> None:
        """Wait for background workers before closing the application."""
        self._import_in_progress = False
        self._stop_worker(self._import_worker, timeout_ms)
        self._import_worker = None
        self._stop_worker(self._cluster_worker, timeout_ms)
        self._cluster_worker = None
        for worker in list(self._fit_workers.values()):
            self._stop_worker(worker, timeout_ms)
        self._fit_workers.clear()
        self.cancel_bulk_fit()
        self._stop_worker(self._bulk_fit_worker, timeout_ms)
        self._bulk_fit_worker = None
        self.cancel_replenishment_qmin_batch()
        self._stop_worker(self._replenishment_qmin_worker, timeout_ms)
        self._replenishment_qmin_worker = None

    def load_directory(self, path: Path, on_progress=None, on_done=None, on_error=None) -> None:
        if self._import_in_progress:
            return
        self.state.data_dir = path
        worker = ImportWorker(path)
        self._import_in_progress = True

        def _clear_import_flag() -> None:
            self._import_in_progress = False
            self._import_worker = None

        if on_progress:
            worker.progress.connect(on_progress)
        if on_done:
            def _done(payload) -> None:
                try:
                    on_done(payload)
                finally:
                    _clear_import_flag()
            worker.finished_ok.connect(_done)
        else:
            worker.finished_ok.connect(lambda _: _clear_import_flag())
        if on_error:
            def _err(message: str) -> None:
                _clear_import_flag()
                on_error(message)
            worker.finished_err.connect(_err)
        else:
            worker.finished_err.connect(lambda _: _clear_import_flag())

        self._start_worker(worker, self._import_worker)
        self._import_worker = worker

    def handle_import_result(self, payload) -> None:
        result, series, desc = payload
        self.state.import_result = result
        self.state.hourly_series = series
        self.state.descriptive_df = desc
        self.state.treated_series = {}
        self.state.outlier_results = {}
        self.state.fit_results = {}
        self.state.replenishment_qmin_results = {}
        self.state.session_selected_skus = set()
        self.state.session_sku_order = []
        self.state.cluster_result = None
        self.state.cluster_map = {}
        self.state.feature_df = build_feature_matrix(series)
        baixo, alto = suggest_threshold_cutoffs(
            self.state.feature_df,
            field=self.state.cluster_cutoff_field,
        )
        self.state.cluster_baixo_cutoff = baixo
        self.state.cluster_alto_cutoff = alto
        self.state.last_updated = datetime.now()
        start, end, n_hours = global_period_bounds(result.normalized_df)
        self.state.global_period_start = start
        self.state.global_period_end = end
        self.state.global_n_hours = n_hours
        if not self.state.selected_sku and series:
            self.select_sku(sorted(series.keys())[0])
        self.state.data_loaded.emit()

    def apply_threshold_clusters(
        self,
        alto_cutoff: float | None = None,
        baixo_cutoff: float | None = None,
        cutoff_field: str | None = None,
    ) -> ClusterResult:
        """Apply demand/h cutoffs synchronously — preview and result always match."""
        alto = alto_cutoff if alto_cutoff is not None else self.state.cluster_alto_cutoff
        baixo = baixo_cutoff if baixo_cutoff is not None else self.state.cluster_baixo_cutoff
        field = cutoff_field or self.state.cluster_cutoff_field

        self.state.cluster_alto_cutoff = alto
        self.state.cluster_baixo_cutoff = baixo
        self.state.cluster_cutoff_field = field

        if self.state.feature_df is None or self.state.feature_df.empty:
            self.state.feature_df = build_feature_matrix(self.state.hourly_series)

        result = run_threshold_clustering(
            self.state.feature_df,
            alto_cutoff=alto,
            baixo_cutoff=baixo,
            cutoff_field=field,
        )
        self.handle_cluster_result(result)
        return result

    def run_clustering(
        self,
        on_progress=None,
        on_done=None,
        on_error=None,
        *,
        alto_cutoff: float | None = None,
        baixo_cutoff: float | None = None,
        cutoff_field: str | None = None,
    ) -> None:
        alto = alto_cutoff if alto_cutoff is not None else self.state.cluster_alto_cutoff
        baixo = baixo_cutoff if baixo_cutoff is not None else self.state.cluster_baixo_cutoff
        field = cutoff_field or self.state.cluster_cutoff_field
        if self.state.clustering_method != "kmeans":
            try:
                result = self.apply_threshold_clusters(alto, baixo, field)
                if on_done:
                    on_done(result)
            except ValueError as exc:
                if on_error:
                    on_error(str(exc))
                else:
                    self.show_error(str(exc))
            return

        self.state.cluster_alto_cutoff = alto
        self.state.cluster_baixo_cutoff = baixo
        self.state.cluster_cutoff_field = field

        worker = ClusterWorker(self.state, alto, baixo, field)
        if on_progress:
            worker.progress.connect(on_progress)
        if on_done:
            worker.finished_ok.connect(on_done)
        if on_error:
            worker.finished_err.connect(on_error)
        self._start_worker(worker, self._cluster_worker)
        self._cluster_worker = worker

    def handle_cluster_result(self, result) -> None:
        self.state.cluster_result = result
        self.state.cluster_map = {a.produto: a for a in result.assignments}
        if self.state.descriptive_df is not None and not self.state.descriptive_df.empty:
            self.state.descriptive_df["cluster"] = self.state.descriptive_df["produto"].map(
                lambda p: self.state.cluster_map.get(str(p), None) and self.state.cluster_map[str(p)].cluster_name or ""
            )
        self.state.clusters_updated.emit()

    def apply_outliers(self, produto: str, method: str, params: dict | None = None) -> None:
        hs = self.state.hourly_series.get(produto)
        if hs is None:
            return
        result = apply_outlier_method(hs, method, params)
        self.state.outlier_results[produto] = result
        self.state.treated_series[produto] = apply_treatment(hs, result)
        self.state.outliers_changed.emit()

    def restore_outliers(self, produto: str) -> None:
        self.state.outlier_results.pop(produto, None)
        self.state.treated_series.pop(produto, None)
        self.state.outliers_changed.emit()

    @staticmethod
    def _fit_cache_key(produto: str, positive_only: bool, value_mode: str) -> str:
        zero_tag = "pos_only" if positive_only else "with_zeros"
        return f"{produto}__{value_mode}__{zero_tag}"

    def get_fit_results(
        self,
        produto: str,
        positive_only: bool = False,
        value_mode: str | None = None,
    ) -> list | None:
        mode = value_mode or self.state.demand_value_mode
        return self.state.fit_results.get(self._fit_cache_key(produto, positive_only, mode))

    def is_fitting(
        self,
        produto: str,
        positive_only: bool = False,
        value_mode: str | None = None,
    ) -> bool:
        mode = value_mode or self.state.demand_value_mode
        key = self._fit_cache_key(produto, positive_only, mode)
        worker = self._fit_workers.get(key)
        return worker is not None and worker.isRunning()

    def request_fit(
        self,
        produto: str,
        positive_only: bool = False,
        value_mode: str | None = None,
    ) -> list | None:
        """Return cached fits, start background fit, or None while pending."""
        hs = self.state.get_active_series(produto) or self.state.hourly_series.get(produto)
        if hs is None:
            return []
        values = prepare_demand_values(
            hs.quantities.values.astype(float),
            include_zeros=not positive_only,
        )
        if len(values) == 0:
            return []
        mode = value_mode or self.state.demand_value_mode
        return self.request_fit_values(
            produto, values, positive_only=positive_only, value_mode=mode,
        )

    def request_fit_values(
        self,
        cache_key: str,
        values: np.ndarray,
        *,
        positive_only: bool = False,
        value_mode: str | None = None,
    ) -> list | None:
        """Fit arbitrary value array in background (SKU or cluster aggregate)."""
        mode = value_mode or self.state.demand_value_mode
        key = self._fit_cache_key(cache_key, positive_only, mode)
        cached = self.state.fit_results.get(key)
        if cached is not None:
            return cached
        if self.is_fitting(cache_key, positive_only, mode):
            return None

        worker = FitWorker(
            cache_key,
            np.asarray(values, dtype=float),
            include_zeros=not positive_only,
            value_mode=mode,
        )
        worker.finished_ok.connect(
            lambda sku, results, po=positive_only, vm=mode: self._on_fit_done(sku, po, vm, results)
        )
        worker.finished_err.connect(
            lambda sku, message: self.show_error(f"Ajuste falhou ({sku}): {message}")
        )
        fit_key = self._fit_cache_key(cache_key, positive_only, mode)
        previous = self._fit_workers.get(fit_key)
        self._start_worker(worker, previous)
        self._fit_workers[fit_key] = worker
        return None

    def _on_fit_done(
        self,
        produto: str,
        positive_only: bool,
        value_mode: str,
        results: list,
    ) -> None:
        cache_key = self._fit_cache_key(produto, positive_only, value_mode)
        self.state.fit_results[cache_key] = results
        self._fit_workers.pop(cache_key, None)
        self.state.fit_completed.emit(produto)

    def is_bulk_fitting(self) -> bool:
        return self._bulk_fit_worker is not None and self._bulk_fit_worker.isRunning()

    def cancel_bulk_fit(self) -> None:
        self._bulk_fit_cancel = True
        if self._bulk_fit_worker is not None and self._bulk_fit_worker.isRunning():
            self._bulk_fit_worker.requestInterruption()

    def start_bulk_fit_all_skus(
        self,
        *,
        use_treated: bool = False,
        on_progress=None,
        on_sku_done=None,
        on_sku_failed=None,
        on_done=None,
        on_error=None,
    ) -> int:
        """Fit all SKUs in background; returns number of SKUs queued (excluding cache)."""
        if self.is_bulk_fitting():
            return 0

        mode = self.state.demand_value_mode
        series = dict(self.state.hourly_series)
        if use_treated:
            series.update(self.state.treated_series)

        jobs: list[tuple[str, np.ndarray]] = []
        in_scope = set(self.state.processing_skus())
        for produto, hs in sorted(series.items()):
            if produto not in in_scope:
                continue
            if self.get_fit_results(produto, positive_only=False, value_mode=mode) is not None:
                continue
            values = prepare_demand_values(
                hs.quantities.values.astype(float),
                include_zeros=True,
            )
            if len(values) == 0:
                continue
            jobs.append((produto, values))

        if not jobs:
            if on_done:
                on_done()
            return 0

        self._bulk_fit_cancel = False
        worker = BulkFitWorker(jobs, include_zeros=True, value_mode=mode)
        if on_progress:
            worker.progress.connect(on_progress)
        if on_sku_done:
            worker.sku_done.connect(
                lambda sku, results: self._on_bulk_sku_done(sku, results, on_sku_done)
            )
        if on_sku_failed:
            worker.sku_failed.connect(on_sku_failed)
        if on_done:
            worker.finished_ok.connect(on_done)
        if on_error:
            worker.finished_err.connect(on_error)

        worker.set_cancel_check(lambda: self._bulk_fit_cancel)

        def _clear_bulk() -> None:
            self._bulk_fit_worker = None

        worker.finished.connect(_clear_bulk)
        self._start_worker(worker, self._bulk_fit_worker)
        self._bulk_fit_worker = worker
        return len(jobs)

    def _on_bulk_sku_done(self, produto: str, results: list, callback) -> None:
        cache_key = self._fit_cache_key(produto, False, self.state.demand_value_mode)
        self.state.fit_results[cache_key] = results
        callback(produto, results)

    def is_replenishment_qmin_running(self) -> bool:
        return (
            self._replenishment_qmin_worker is not None
            and self._replenishment_qmin_worker.isRunning()
        )

    def cancel_replenishment_qmin_batch(self) -> None:
        self._replenishment_qmin_cancel = True
        if (
            self._replenishment_qmin_worker is not None
            and self._replenishment_qmin_worker.isRunning()
        ):
            self._replenishment_qmin_worker.requestInterruption()

    def start_replenishment_qmin_batch(
        self,
        *,
        replenishment_minutes: float = 120.0,
        n_trajetorias: int = 100,
        on_progress=None,
        on_sku_done=None,
        on_sku_failed=None,
        on_done=None,
        on_error=None,
    ) -> int:
        """Simulate replenishment Qmin for session-selected SKUs. Returns job count."""
        if self.is_replenishment_qmin_running():
            return 0

        mode = self.state.demand_value_mode
        jobs: list[tuple[str, object]] = []
        for produto in self.state.session_sku_order:
            if produto not in self.state.session_selected_skus:
                continue
            fits = self.get_fit_results(produto, positive_only=False, value_mode=mode)
            fit = best_fit(fits or [])
            if fit is None:
                if on_sku_failed:
                    on_sku_failed(produto, "SKU sem ajuste convergente.")
                continue
            jobs.append((produto, fit))

        if not jobs:
            if on_done:
                on_done()
            return 0

        self._replenishment_qmin_cancel = False
        worker = ReplenishmentQminWorker(
            jobs,
            replenishment_minutes=replenishment_minutes,
            n_trajetorias=n_trajetorias,
            seed=self.state.config.random_state,
        )
        if on_progress:
            worker.progress.connect(on_progress)
        if on_sku_done:
            worker.sku_done.connect(
                lambda sku, result: self._on_replenishment_sku_done(sku, result, on_sku_done)
            )
        if on_sku_failed:
            worker.sku_failed.connect(on_sku_failed)
        if on_done:
            worker.finished_ok.connect(on_done)
        if on_error:
            worker.finished_err.connect(on_error)

        worker.set_cancel_check(lambda: self._replenishment_qmin_cancel)

        def _clear_worker() -> None:
            self._replenishment_qmin_worker = None

        worker.finished.connect(_clear_worker)
        self._start_worker(worker, self._replenishment_qmin_worker)
        self._replenishment_qmin_worker = worker
        return len(jobs)

    def _on_replenishment_sku_done(self, produto: str, result, callback) -> None:
        self.state.replenishment_qmin_results[produto] = result
        callback(produto, result)

    def fit_sku(
        self,
        produto: str,
        include_zeros: bool = True,
        positive_only: bool = False,
        value_mode: str | None = None,
    ) -> list:
        """Synchronous fit — prefer request_fit() from UI threads."""
        mode = value_mode or self.state.demand_value_mode
        cached = self.get_fit_results(produto, positive_only, mode)
        if cached is not None:
            return cached
        hs = self.state.get_active_series(produto) or self.state.hourly_series.get(produto)
        if hs is None:
            return []
        values = prepare_demand_values(
            hs.quantities.values.astype(float),
            include_zeros=include_zeros and not positive_only,
        )
        if len(values) == 0:
            return []
        results = fit_all(values, include_zeros=include_zeros, value_mode=mode)
        cache_key = self._fit_cache_key(produto, positive_only, mode)
        self.state.fit_results[cache_key] = results
        self.state.fit_completed.emit(produto)
        return results

    def clear_cache(self) -> None:
        self.cache.clear()

    def select_sku(self, produto: str) -> None:
        produto = str(produto)
        if produto and produto not in self.state.session_selected_skus:
            self.state.session_selected_skus.add(produto)
            if produto not in self.state.session_sku_order:
                self.state.session_sku_order.append(produto)
            self.state.session_skus_changed.emit()
        if self.state.selected_sku == produto:
            return
        self.state.selected_sku = produto
        self.state.sku_changed.emit(produto)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self.parent, "Erro", message)
