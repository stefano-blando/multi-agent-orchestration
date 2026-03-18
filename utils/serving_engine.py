"""Serving decision planner and deterministic executor helpers."""

from __future__ import annotations

from dataclasses import dataclass

from utils.serving_utils import dish_has_intolerance, extract_intolerances


@dataclass
class ServingTask:
    meal_id: str
    customer_id: str
    client_name: str
    request_text: str
    requested_dish: str
    dish_to_serve: str
    action: str  # "serve" | "skip"
    reason: str = ""


def _extract_requested_dish_name(state, request_text: str) -> str:
    extract_fn = getattr(state, "_extract_requested_dish", None)
    if callable(extract_fn):
        out = str(extract_fn(request_text) or "")
        if out:
            return out
    return ""


def _can_cook_with(recipe: dict, remaining: dict[str, int]) -> bool:
    """Check if recipe is cookable with remaining inventory."""
    for ing, qty in recipe.get("ingredients", {}).items():
        if remaining.get(ing, 0) < int(qty):
            return False
    return True


def _deduct_ingredients(recipe: dict, remaining: dict[str, int]) -> None:
    """Deduct recipe ingredients from remaining inventory (in-place)."""
    for ing, qty in recipe.get("ingredients", {}).items():
        remaining[ing] = remaining.get(ing, 0) - int(qty)


def build_serving_plan(state, meals: list[dict], recipes: list[dict]) -> list[ServingTask]:
    """Build serving plan sorted by value, tracking inventory as we assign."""
    waiting = [
        m
        for m in meals
        if not bool(m.get("executed")) and str(m.get("status", "")).strip().lower() == "waiting"
    ]

    by_name = {str(r.get("name", "")): r for r in recipes if str(r.get("name", ""))}

    # Enrich each meal with requested dish + prestige for sorting.
    enriched: list[tuple[dict, str, int]] = []
    for meal in waiting:
        request_text = str(meal.get("orderText") or meal.get("request") or "")
        requested = _extract_requested_dish_name(state, request_text)
        recipe = by_name.get(requested) if requested else None
        prestige = int(recipe.get("prestige", 0)) if recipe else 0
        enriched.append((meal, requested, prestige))

    # Sort: highest prestige first, then earliest startTime as tiebreaker.
    enriched.sort(key=lambda x: (-x[2], str(x[0].get("startTime", ""))))

    remaining = dict(state.inventory)
    tasks: list[ServingTask] = []

    for meal, requested, prestige in enriched:
        meal_id = str(meal.get("id", "")).strip()
        customer_id = str(meal.get("customerId", "")).strip()
        request_text = str(meal.get("orderText") or meal.get("request") or "")
        client_name = str(meal.get("clientName") or (meal.get("customer") or {}).get("name") or "unknown")

        if not requested:
            tasks.append(
                ServingTask(
                    meal_id=meal_id,
                    customer_id=customer_id,
                    client_name=client_name,
                    request_text=request_text,
                    requested_dish="",
                    dish_to_serve="",
                    action="skip",
                    reason="requested_dish_not_recognized",
                )
            )
            continue

        recipe = by_name.get(requested)
        if not recipe or not _can_cook_with(recipe, remaining):
            tasks.append(
                ServingTask(
                    meal_id=meal_id,
                    customer_id=customer_id,
                    client_name=client_name,
                    request_text=request_text,
                    requested_dish=requested,
                    dish_to_serve=requested,
                    action="skip",
                    reason="requested_dish_not_cookable",
                )
            )
            continue

        intolerances = extract_intolerances(request_text)
        if intolerances and dish_has_intolerance(recipe, intolerances):
            tasks.append(
                ServingTask(
                    meal_id=meal_id,
                    customer_id=customer_id,
                    client_name=client_name,
                    request_text=request_text,
                    requested_dish=requested,
                    dish_to_serve=requested,
                    action="skip",
                    reason="intolerance_conflict",
                )
            )
            continue

        # Deduct ingredients so subsequent dishes check against real remaining stock.
        _deduct_ingredients(recipe, remaining)
        tasks.append(
            ServingTask(
                meal_id=meal_id,
                customer_id=customer_id,
                client_name=client_name,
                request_text=request_text,
                requested_dish=requested,
                dish_to_serve=requested,
                action="serve",
                reason="ok",
            )
        )

    return tasks
