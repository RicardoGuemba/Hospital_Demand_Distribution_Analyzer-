"""Data import view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from hospital_demand_analyzer.app.controller import AppController
from hospital_demand_analyzer.app.state import AppState
from hospital_demand_analyzer.ui.nav import STEP_DATA
from hospital_demand_analyzer.ui.widgets.workflow_bar import WorkflowBar


class StatCard(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 11px; color: #666;")
        self.value_label = QLabel("—")
        self.value_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class DataView(QWidget):
    def __init__(self, state: AppState, controller: AppController, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.controller = controller
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(WorkflowBar(STEP_DATA))

        header = QHBoxLayout()
        self.path_label = QLabel(str(self.state.data_dir))
        btn_select = QPushButton("Selecionar Pasta")
        btn_reload = QPushButton("Recarregar Dados")
        btn_clear = QPushButton("Limpar Cache")
        btn_select.clicked.connect(self._select_folder)
        btn_reload.clicked.connect(self._reload)
        btn_clear.clicked.connect(self._clear_cache)
        header.addWidget(QLabel("Pasta:"))
        header.addWidget(self.path_label, 1)
        header.addWidget(btn_select)
        header.addWidget(btn_reload)
        header.addWidget(btn_clear)
        layout.addLayout(header)

        cards = QGridLayout()
        self.card_files = StatCard("Arquivos")
        self.card_valid = StatCard("Registros Válidos")
        self.card_rejected = StatCard("Rejeitados")
        self.card_skus = StatCard("SKUs Únicos")
        self.card_start = StatCard("Data Inicial")
        self.card_end = StatCard("Data Final")
        for i, c in enumerate([self.card_files, self.card_valid, self.card_rejected,
                                self.card_skus, self.card_start, self.card_end]):
            cards.addWidget(c, 0, i)
        layout.addLayout(cards)

        self.files_table = QTableWidget(0, 7)
        self.files_table.setHorizontalHeaderLabels(
            ["Arquivo", "Linhas", "SKUs", "Data Inicial", "Data Final", "Erros", "Status"]
        )
        self.files_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.files_table)

        inc_group = QGroupBox("Inconsistências")
        inc_layout = QVBoxLayout(inc_group)
        self.inc_label = QLabel("Nenhuma inconsistência detectada.")
        self.inc_label.setWordWrap(True)
        inc_layout.addWidget(self.inc_label)
        layout.addWidget(inc_group)

    def populate_light(self) -> None:
        self.refresh()

    def _select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Dados")
        if folder:
            self.path_label.setText(folder)
            self._load(folder)

    def _reload(self) -> None:
        self._load(str(self.state.data_dir))

    def _load(self, path: str) -> None:
        from pathlib import Path

        def _on_done(payload) -> None:
            self.controller.handle_import_result(payload)
            self.refresh()

        self.controller.load_directory(
            Path(path),
            on_progress=lambda msg: self.path_label.setToolTip(msg),
            on_done=_on_done,
            on_error=self.controller.show_error,
        )

    def _clear_cache(self) -> None:
        self.controller.clear_cache()

    def refresh(self) -> None:
        r = self.state.import_result
        if r is None:
            return
        self.path_label.setText(r.data_dir)
        self.card_files.set_value(str(r.total_files))
        self.card_valid.set_value(str(r.total_valid_rows))
        self.card_rejected.set_value(str(r.total_rejected))
        self.card_skus.set_value(str(r.unique_skus))
        self.card_start.set_value(r.date_min.strftime("%Y-%m-%d %H:%M") if r.date_min else "—")
        self.card_end.set_value(r.date_max.strftime("%Y-%m-%d %H:%M") if r.date_max else "—")

        self.files_table.setRowCount(len(r.file_reports))
        for i, fr in enumerate(r.file_reports):
            self.files_table.setItem(i, 0, QTableWidgetItem(fr.file_name))
            self.files_table.setItem(i, 1, QTableWidgetItem(str(fr.rows_valid)))
            self.files_table.setItem(i, 2, QTableWidgetItem(str(fr.skus)))
            self.files_table.setItem(i, 3, QTableWidgetItem(
                fr.date_min.strftime("%Y-%m-%d") if fr.date_min else "—"))
            self.files_table.setItem(i, 4, QTableWidgetItem(
                fr.date_max.strftime("%Y-%m-%d") if fr.date_max else "—"))
            self.files_table.setItem(i, 5, QTableWidgetItem(str(fr.error_count)))
            self.files_table.setItem(i, 6, QTableWidgetItem(fr.status))

        inc = r.inconsistencies
        lines = []
        if inc.missing_columns:
            lines.append(f"Colunas ausentes: {', '.join(inc.missing_columns)}")
        if inc.invalid_dates:
            lines.append(f"Datas inválidas: {inc.invalid_dates}")
        if inc.invalid_quantities:
            lines.append(f"Quantidades inválidas: {inc.invalid_quantities}")
        if inc.skus_without_description:
            lines.append(f"SKUs sem descrição: {len(inc.skus_without_description)}")
        if inc.divergent_descriptions:
            lines.append(f"Descrições divergentes: {len(inc.divergent_descriptions)} SKUs")
        if inc.duplicate_rows:
            lines.append(f"Linhas duplicadas: {inc.duplicate_rows}")
        self.inc_label.setText("\n".join(lines) if lines else "Nenhuma inconsistência detectada.")
