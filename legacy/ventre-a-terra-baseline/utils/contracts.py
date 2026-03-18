"""Shared typed contracts for strategy decisions."""

from __future__ import annotations

from typing import TypedDict


class MenuItem(TypedDict):
    name: str
    price: int


class BidEntry(TypedDict):
    ingredient: str
    bid: int
    quantity: int
