"""Export view."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.exports.exporter import (
    export_cluster_results,
    export_csv,
    export_excel,
    export_fit_ranking,
    export_time_series,
)
from hospital_demand_analyzer.services.time_series import hourly_series_to_dataframe
from hospital_demand_analyzer.ui.nav import STEP_REPLENISHMENT_QMIN, STEP_SKU_CDF
from hospital_demand_analyzer.utils.paths import project_root


class ExportView(QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Exportar resultados para CSV ou Excel."))

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Formato:"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["csv", "xlsx"])
        fmt_row.addWidget(self.fmt_combo)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        for label, handler in [
            ("Exportar Séries Temporais", self._export_series),
            ("Exportar Clusters", self._export_clusters),
            ("Exportar Ranking SKU Selecionado", self._export_ranking),
            ("Exportar Consolidado por SKU", self._export_consolidated),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            layout.addWidget(btn)
        btn_cdf = QPushButton("Ir para CDF por SKU")
        btn_cdf.clicked.connect(self._go_sku_cdf)
        layout.addWidget(btn_cdf)
        btn_qmin = QPushButton("Ir para Qmin reabastecimento")
        btn_qmin.clicked.connect(self._go_replenishment_qmin)
        layout.addWidget(btn_qmin)
        layout.addStretch()

    def _go_sku_cdf(self) -> None:
        win = self.window()
        if hasattr(win, "navigate_to"):
            win.navigate_to(STEP_SKU_CDF)

    def _go_replenishment_qmin(self) -> None:
        win = self.window()
        if hasattr(win, "navigate_to"):
            win.navigate_to(STEP_REPLENISHMENT_QMIN)

    def _fmt(self) -> str:
        return self.fmt_combo.currentText()

    def _save_path(self, default_name: str) -> Path | None:
        fmt = self._fmt()
        if fmt == "xlsx":
            path, _ = QFileDialog.getSaveFileName(self, "Salvar", default_name, "Excel (*.xlsx)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Salvar", default_name, "CSV (*.csv)")
        return Path(path) if path else None

    def _export_series(self) -> None:
        if not self.state.hourly_series:
            return
        path = self._save_path(str(project_root() / "export_series.csv"))
        if path:
            df = hourly_series_to_dataframe(self.state.hourly_series)
            df["cluster"] = df["produto"].map(
                lambda p: self.state.get_cluster_for_sku(str(p)))
            export_time_series(df, path, self._fmt())

    def _export_clusters(self) -> None:
        if not self.state.cluster_result:
            return
        path = self._save_path(str(project_root() / "export_clusters.csv"))
        if path:
            export_cluster_results(self.state.cluster_result.assignments, path, self._fmt())

    def _export_ranking(self) -> None:
        sku = self.state.selected_sku
        if not sku or sku not in self.state.fit_results:
            return
        path = self._save_path(str(project_root() / f"export_ranking_{sku}.csv"))
        if path:
            export_fit_ranking(self.state.fit_results[sku], path, self._fmt())

    def _export_consolidated(self) -> None:
        if self.state.descriptive_df.empty:
            return
        path = self._save_path(str(project_root() / "export_consolidado.csv"))
        if not path:
            return
        df = self.state.descriptive_df.copy()
        df["cluster"] = df["produto"].astype(str).map(self.state.get_cluster_for_sku)
        rows = []
        for _, row in df.iterrows():
            sku = str(row["produto"])
            fits = self.state.fit_results.get(sku, [])
            top2 = sorted([f for f in fits if f.converged], key=lambda f: f.aic)[:2]
            outlier = self.state.outlier_results.get(sku)
            rows.append({
                "Produto": sku,
                "Descricao": row.get("descricao", ""),
                "Cluster": row.get("cluster", ""),
                "N": row.get("n", 0),
                "% zeros": f"{row.get('zero_ratio', 0)*100:.1f}",
                "Media": row.get("mean", 0),
                "P95": row.get("p95", 0),
                "P99": row.get("p99", 0),
                "Melhor Distribuicao": top2[0].name if top2 else "",
                "Segunda Distribuicao": top2[1].name if len(top2) > 1 else "",
                "AIC #1": top2[0].aic if top2 else "",
                "BIC #1": top2[0].bic if top2 else "",
                "Tail Score #1": top2[0].tail_fit_score if top2 else "",
                "Metodo Outlier": outlier.method if outlier else "",
                "% Outlier": f"{outlier.outlier_ratio*100:.1f}" if outlier else "0",
            })
        import pandas as pd
        out_df = pd.DataFrame(rows)
        if self._fmt() == "xlsx":
            export_excel(out_df, path, "Consolidado")
        else:
            export_csv(out_df, path)
