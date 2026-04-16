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
