"""Background workers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.clustering.features import build_feature_matrix
from hospital_demand_analyzer.clustering.kmeans_service import run_kmeans
from hospital_demand_analyzer.clustering.threshold_clustering import run_threshold_clustering
from hospital_demand_analyzer.data.importer import import_directory
from hospital_demand_analyzer.services.time_series import build_all_hourly_series
from hospital_demand_analyzer.statistics.descriptive import compute_all_descriptive
from hospital_demand_analyzer.statistics.fitter import fit_all


class ImportWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(object)
    finished_err = Signal(str)

    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self.data_dir = data_dir

    def run(self) -> None:
        try:
            self.progress.emit("Importando arquivos...")
            result = import_directory(self.data_dir)
            self.progress.emit("Construindo séries horárias...")
            series = build_all_hourly_series(result.normalized_df)
            desc = compute_all_descriptive(series)
            self.finished_ok.emit((result, series, desc))
        except Exception as exc:
            self.finished_err.emit(str(exc))


class ClusterWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(object)
    finished_err = Signal(str)

    def __init__(
        self,
        state: AppState,
        alto_cutoff: float,
        baixo_cutoff: float,
        cutoff_field: str,
    ) -> None:
        super().__init__()
        self.state = state
        self.alto_cutoff = alto_cutoff
        self.baixo_cutoff = baixo_cutoff
        self.cutoff_field = cutoff_field

    def run(self) -> None:
        try:
            self.progress.emit("Calculando features...")
            if self.state.feature_df is not None and not self.state.feature_df.empty:
                feat_df = self.state.feature_df
            else:
                series = self.state.hourly_series
                if self.state.use_treated and self.state.treated_series:
                    series = {**self.state.hourly_series, **self.state.treated_series}
                feat_df = build_feature_matrix(series)
            if self.state.clustering_method == "kmeans":
                self.progress.emit("Executando K-Means...")
                result = run_kmeans(
                    feat_df,
                    random_state=self.state.config.random_state,
                    n_clusters=self.state.config.kmeans_n_clusters,
                    scaler_name=self.state.config.default_scaler,
                )
            else:
                self.progress.emit("Aplicando cortes de demanda/h...")
                result = run_threshold_clustering(
                    feat_df,
                    alto_cutoff=self.alto_cutoff,
                    baixo_cutoff=self.baixo_cutoff,
                    cutoff_field=self.cutoff_field,
                )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.finished_err.emit(str(exc))


class FitWorker(QThread):
    progress = Signal(str, int, int)
    finished_ok = Signal(str, object)
    finished_err = Signal(str, str)

    def __init__(
        self,
        produto: str,
        values,
        include_zeros: bool = True,
        value_mode: str = "discrete",
    ) -> None:
        super().__init__()
        self.produto = produto
        self.values = values
        self.include_zeros = include_zeros
        self.value_mode = value_mode

    def run(self) -> None:
        try:
            self.progress.emit(self.produto, 0, 1)
            results = fit_all(
                self.values,
                include_zeros=self.include_zeros,
                value_mode=self.value_mode,
            )
            self.finished_ok.emit(self.produto, results)
        except Exception as exc:
            self.finished_err.emit(self.produto, str(exc))


class BulkFitWorker(QThread):
    progress = Signal(str, int, int)
    sku_done = Signal(str, object)
    sku_failed = Signal(str, str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        jobs: list[tuple[str, np.ndarray]],
        *,
        include_zeros: bool = True,
        value_mode: str = "discrete",
    ) -> None:
        super().__init__()
        self.jobs = jobs
        self.include_zeros = include_zeros
        self.value_mode = value_mode
        self._cancel_check = None

    def set_cancel_check(self, fn) -> None:
        self._cancel_check = fn

    def run(self) -> None:
        try:
            total = len(self.jobs)
            for i, (produto, values) in enumerate(self.jobs):
                if self.isInterruptionRequested() or (self._cancel_check and self._cancel_check()):
                    break
                self.progress.emit(produto, i, total)
                try:
                    results = fit_all(
                        values,
                        include_zeros=self.include_zeros,
                        value_mode=self.value_mode,
                    )
                    self.sku_done.emit(produto, results)
                except Exception as exc:
                    self.sku_failed.emit(produto, str(exc))
            self.progress.emit("", total, total)
            self.finished_ok.emit()
        except Exception as exc:
            self.finished_err.emit(str(exc))


class ReplenishmentQminWorker(QThread):
    progress = Signal(str, int, int)
    sku_done = Signal(str, object)
    sku_failed = Signal(str, str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        jobs: list[tuple[str, object]],
        *,
        replenishment_minutes: float,
        n_trajetorias: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.jobs = jobs
        self.replenishment_minutes = replenishment_minutes
        self.n_trajetorias = n_trajetorias
        self.seed = seed
        self._cancel_check = None

    def set_cancel_check(self, fn) -> None:
        self._cancel_check = fn

    def run(self) -> None:
        try:
            from hospital_demand_analyzer.statistics.replenishment_qmin import (
                run_replenishment_qmin_monte_carlo,
            )

            total = len(self.jobs)
            for i, (produto, fit) in enumerate(self.jobs):
                if self.isInterruptionRequested() or (self._cancel_check and self._cancel_check()):
                    break
                self.progress.emit(produto, i, total)
                try:
                    sku_seed = None if self.seed is None else self.seed + i
                    result = run_replenishment_qmin_monte_carlo(
                        produto,
                        self.replenishment_minutes,
                        fit,
                        n_trajetorias=self.n_trajetorias,
                        seed=sku_seed,
                    )
                    self.sku_done.emit(produto, result)
                except Exception as exc:
                    self.sku_failed.emit(produto, str(exc))
            self.progress.emit("", total, total)
            self.finished_ok.emit()
        except Exception as exc:
            self.finished_err.emit(str(exc))


class MonteCarloWorker(QThread):
    finished_ok = Signal(object)
    finished_err = Signal(str)

    def __init__(
        self,
        produto: str,
        saldo_inicial: float,
        qmin: float,
        fit,
        n_trajetorias: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.produto = produto
        self.saldo_inicial = saldo_inicial
        self.qmin = qmin
        self.fit = fit
        self.n_trajetorias = n_trajetorias
        self.seed = seed

    def run(self) -> None:
        try:
            from hospital_demand_analyzer.statistics.monte_carlo import run_tempqmin_monte_carlo

            result = run_tempqmin_monte_carlo(
                self.produto,
                self.saldo_inicial,
                self.qmin,
                self.fit,
                n_trajetorias=self.n_trajetorias,
                seed=self.seed,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.finished_err.emit(str(exc))
