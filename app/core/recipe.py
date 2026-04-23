from __future__ import annotations

import json
from pathlib import Path

from app.core.models import Recipe


class RecipeLoader:
    @staticmethod
    def load(path: str | Path) -> Recipe:
        recipe_path = Path(path)
        with recipe_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return Recipe.model_validate(payload)

    @staticmethod
    def discover(directory: str | Path) -> list[tuple[Path, Recipe]]:
        recipe_dir = Path(directory)
        recipes: list[tuple[Path, Recipe]] = []
        if not recipe_dir.exists():
            return recipes
        for recipe_path in sorted(recipe_dir.glob("*.json")):
            try:
                recipes.append((recipe_path, RecipeLoader.load(recipe_path)))
            except Exception:
                continue
        return recipes
