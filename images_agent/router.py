from __future__ import annotations

import enum


class RoutedTask(enum.Enum):
    DUPLICATES = "dupes"
    SORT_AGE = "sort-age"
    AMBIGUOUS = "ambiguous"


_DUPE_KEYWORDS = frozenset(
    {"duplicate", "duplicates", "dupe", "dupes", "same file", "identical", "byte-identical", "exact duplicate"}
)

_AGE_KEYWORDS = frozenset(
    {"age", "aged", "child", "children", "kid", "kids", "older", "younger", "sort", "order", "ordering"}
)


def route_task(text: str) -> RoutedTask:
    lower = text.lower()
    dupe_hit = any(k in lower for k in _DUPE_KEYWORDS)
    age_hit = any(k in lower for k in _AGE_KEYWORDS)

    if dupe_hit and not age_hit:
        return RoutedTask.DUPLICATES
    if age_hit and not dupe_hit:
        return RoutedTask.SORT_AGE
    if dupe_hit and age_hit:
        return RoutedTask.AMBIGUOUS
    return RoutedTask.AMBIGUOUS


def ambiguous_hint() -> str:
    return (
        "Could not infer the task from that text.\n"
        "Try mentioning duplicates (e.g. duplicated, identical files) "
        "or sorting by child age (e.g. sort photos by child age).\n"
        "Or run: images-agent dupes   /   images-agent sort-age\n"
    )
