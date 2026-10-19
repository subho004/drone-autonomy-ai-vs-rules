"""Shared pagination envelope builder."""

from __future__ import annotations

from app.schemas.api import Page


def make_page[T](items: list[T], total: int, page: int, page_size: int) -> Page[T]:
    pages = max(1, -(-total // page_size)) if total else 0
    return Page(items=items, total=total, page=page, page_size=page_size, pages=pages)
