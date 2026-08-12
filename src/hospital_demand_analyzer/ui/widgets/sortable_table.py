"""Helpers for sortable product tables (click header → sort)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem


class SortableTableMixin:
    """Sort dataframe-backed tables by column; default descending (maior → menor)."""

    _sort_field: str
    _sort_desc: bool = True
    _sort_column_map: list[str]

    def _init_sortable_table(
        self,
        table: QTableWidget,
        column_fields: list[str],
        *,
        default_field: str,
    ) -> None:
        self._sort_table = table
        self._sort_column_map = column_fields
        self._sort_field = default_field
        self._sort_desc = True
        header = table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._on_sort_header_clicked)
        self._update_sort_indicator()

    def _on_sort_header_clicked(self, column: int) -> None:
        if column < 0 or column >= len(self._sort_column_map):
            return
        field = self._sort_column_map[column]
        if self._sort_field == field:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_field = field
            self._sort_desc = True
        self._update_sort_indicator()
        self._apply_table_sort()

    def _update_sort_indicator(self) -> None:
        if not hasattr(self, "_sort_table"):
            return
        try:
            col = self._sort_column_map.index(self._sort_field)
        except ValueError:
            return
        order = Qt.SortOrder.DescendingOrder if self._sort_desc else Qt.SortOrder.AscendingOrder
        self._sort_table.horizontalHeader().setSortIndicator(col, order)

    def _sort_dataframe(self, df):
        if df is None or df.empty:
            return df
        field = self._sort_field
        if field not in df.columns:
            return df
        ascending = not self._sort_desc
        if field in ("produto", "descricao", "cluster", "cluster_name"):
            return df.sort_values(field, ascending=ascending, key=lambda s: s.astype(str).str.lower())
        return df.sort_values(field, ascending=ascending, na_position="last")

    @staticmethod
    def _numeric_item(text: str, value: float | int) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, float(value))
        return item

    def _apply_table_sort(self) -> None:
        raise NotImplementedError
