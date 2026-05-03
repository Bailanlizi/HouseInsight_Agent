from pathlib import Path


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


class ProjectPaths:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or find_project_root()
        self.data_raw = self.root / "data" / "raw"
        self.data_output = self.root / "data" / "output"
        self.templates = self.root / "templates"

    def raw_dir(self, session_id: str) -> Path:
        d = self.data_raw / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def output_dir(self, session_id: str) -> Path:
        d = self.data_output / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d
