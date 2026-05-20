from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimicry_preproc.pipeline import PipelineConfig
from npbk.trainer import TrainerConfig


@dataclass
class AppConfig:
    store_dir: Path = field(default_factory=lambda: Path.home() / ".mimicry_auth" / "profiles")
    pipeline: PipelineConfig = field(default_factory=PipelineConfig.default)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    n_required_own_samples: int = 11

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        if path is None:
            return cls()
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = Path(path).read_bytes()
        d = tomllib.loads(data.decode())
        return cls(
            store_dir=Path(d.get("store_dir", Path.home() / ".mimicry_auth" / "profiles")),
            pipeline=PipelineConfig.from_dict(d.get("pipeline", {})),
            trainer=TrainerConfig(**d.get("trainer", {})),
            n_required_own_samples=d.get("n_required_own_samples", 11),
        )

    def save(self, path: str | Path) -> None:
        import dataclasses
        import json
        data = {
            "store_dir": str(self.store_dir),
            "n_required_own_samples": self.n_required_own_samples,
            "pipeline": self.pipeline.to_dict(),
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))
