from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    project_dir: str = ""
    video_path: str = ""
    log_path: str = ""
    # Neutral fallbacks.  The web UI replaces both offsets with match_log results
    # for the selected video (for example DJI_0371 is +3 h, +0.4 s).
    utc_offset_hours: float = 0.0
    clock_offset_seconds: float = 0.0
    takeoff_height_m: float = 1.0
    start_seconds: float = 0.0
    end_seconds: float = 10.0
    segments: str = ""
    frame_interval: int = 30
    # FOV without zoom; 0 = camera profile from SRT (Mavic 2 Zoom 4K: 66.1°, see geometry.py).
    horizontal_fov_deg: float = 0.0
    # Correction of FlightRecord heading (camera azimuth = osd.yaw + offset).
    yaw_offset_deg: float = 0.0
    resolution_m_per_px: float = 4.0
    horizon_margin: float = 0.10
    bottom_crop_percent: float = 0.0
    grid_step_m: float = 1000.0
    output_name: str = "mosaic.png"
    run_masking: bool = True
    run_tracking: bool = True
    run_velocities: bool = True
    ice_s_max: int = 110
    ice_v_min: int = 180
    min_ice_area_px: int = 1000
    max_track_distance_m: float = 80.0

    @classmethod
    def from_dict(cls, raw: dict) -> "ProjectConfig":
        allowed = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in allowed}
        # 78.8° was the unverified default of older project.json files; the calibrated profile replaces it.
        if values.get("horizontal_fov_deg") in (78.8, "78.8"):
            values["horizontal_fov_deg"] = 0.0
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def validate(self) -> None:
        if not self.project_dir:
            raise ValueError("Не выбрана папка проекта")
        for label, value in (("Видео", self.video_path), ("DJI-лог", self.log_path)):
            if not value or not Path(value).is_file():
                raise ValueError(f"{label}: файл не найден: {value}")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("Конец фрагмента должен быть позднее его начала")
        if self.frame_interval < 1:
            raise ValueError("Интервал кадров должен быть не меньше 1")
        if self.horizontal_fov_deg and not 1 <= self.horizontal_fov_deg < 179:
            raise ValueError("Горизонтальный FOV должен быть 0 (авто) или между 1° и 179°")
        if not -180 <= self.yaw_offset_deg <= 180:
            raise ValueError("Поправка курса должна быть в пределах ±180°")
        if self.resolution_m_per_px <= 0:
            raise ValueError("Разрешение должно быть положительным")
        if not 0 <= self.bottom_crop_percent <= 95:
            raise ValueError("Нижняя обрезка должна быть от 0 до 95%")

    def time_ranges(self) -> list[tuple[float, float]]:
        if not self.segments.strip():
            return [(self.start_seconds, self.end_seconds)]
        ranges = []
        for item in self.segments.replace("–", "-").replace("—", "-").split(","):
            try:
                start, end = (float(value.strip()) for value in item.split("-", 1))
            except ValueError as exc:
                raise ValueError("Фрагменты задаются как 0-15, 120-135") from exc
            if start < 0 or end <= start:
                raise ValueError("Неверные границы фрагмента")
            ranges.append((start, end))
        return ranges

    def save(self) -> Path:
        self.validate()
        project_dir = Path(self.project_dir).expanduser().resolve()
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / "project.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, ensure_ascii=False, indent=2)
        return path

    def as_dict(self) -> dict:
        return asdict(self)
