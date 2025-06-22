from pathlib import Path

from services.service import Service


class PreprocessingService(Service[Path, Path]):

    def process_item(self, item: Path) -> Path:
        # For now, do nothing but return the item
        return item
