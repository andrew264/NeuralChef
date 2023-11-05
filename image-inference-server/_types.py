from typing import Optional


class ImageResult:
    def __init__(self, class_name: str, description: Optional[str]):
        self.class_name = class_name
        self.description = None if isinstance(description, float) else description

    def __str__(self) -> str:
        out_str = f"Image contains {self.class_name.replace('_', ' ').lower()} "
        if self.description is not None:
            out_str += f"recipe: {self.description}\n"
        return out_str
