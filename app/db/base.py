"""Declarative base for ORM models.

The read-side API uses plain SQL (``sqlalchemy.text``) against the harness
schema, so no ORM models exist yet. SQLite STRICT tables reject ``VARCHAR``,
hence ``str`` maps to ``Text`` here for any future model.
"""

from __future__ import annotations

from typing import ClassVar

from sqlalchemy import Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[type, type[Text]]] = {str: Text}


__all__ = ["Base"]
