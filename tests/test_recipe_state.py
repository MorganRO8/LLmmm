from __future__ import annotations

from app.core.recipe import RecipeLoader
from app.core.state import AppStateStore


def test_advance_step_changes_current_step() -> None:
    recipe = RecipeLoader.load('recipes/scrambled_eggs.json')
    store = AppStateStore(recipe)
    original_step = store.get_current_step().step_id
    result = store.advance_step()
    assert result['current_step']['step_id'] != original_step
    assert original_step in result['completed_step_ids']
