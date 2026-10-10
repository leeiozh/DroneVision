from __future__ import annotations

from dataclasses import asdict
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
import mimetypes
from pathlib import Path
import threading
import traceback
from urllib.parse import parse_qs, quote, urlparse

from .config import ProjectConfig
from .video import probe


STATIC_DIR = Path(__file__).resolve().with_name("web_static")
# Explicit: on Windows mimetypes reads the registry, where .js/.css are sometimes text/plain.
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon"}
STATE = {"status": "idle", "progress": 0.0, "message": "Готов к работе", "output": None, "artifacts": {}, "error": None}
PROJECTION_STATE = {"variants": []}
MOSAIC_STATE = {"result": None}
WAVE_STATE = {"running": False, "progress": 0.0, "message": ""}
LOCK = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _send_media(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range")
        status = 200
        if header and header.startswith("bytes="):
            left, right = header[6:].split(",", 1)[0].split("-", 1)
            if left:
                start = int(left); end = min(int(right) if right else size - 1, size - 1)
            else:  # suffix range: last N bytes
                start = max(size - int(right or 0), 0)
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk: break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def _send_result_png(self, result: dict, saved: bool = False):
        """Send either the saved gridded PNG or a grid-free interactive preview."""
        if saved:
            output = result.get("path")
            if not output or not Path(output).is_file():
                return self._json({"error": "PNG не найден"}, 404)
            data = Path(output).read_bytes()
        else:
            import cv2
            from .adjust import apply_adjustments
            from .cvio import imread

            source = result.get("data_path")
            image = imread(source, cv2.IMREAD_UNCHANGED) if source else None
            if image is None:
                return self._json({"error": "Растр предпросмотра не найден"}, 404)
            image = apply_adjustments(image, result.get("adjustments"))
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                return self._json({"error": "Не удалось собрать предпросмотр"}, 500)
            data = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data); return

    def do_GET(self):
        try:
            return self._get()
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:
            return self._json({"error": str(exc)}, 400)

    def _get(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            with LOCK:
                return self._json(dict(STATE))
        if parsed.path == "/api/catalog":
            from .catalog import interesting_cases
            return self._json(interesting_cases(refresh=parse_qs(parsed.query).get("refresh", ["0"])[0] == "1"))
        if parsed.path == "/api/video":
            from .viewer import proxy_path, safe_media_path
            query = parse_qs(parsed.query)
            source = safe_media_path(query.get("path", [""])[0])
            quality = query.get("quality", ["original"])[0]
            media = proxy_path(str(source), int(quality.rstrip("p"))) if quality != "original" else source
            if not media.is_file():
                return self._json({"error": "Proxy ещё не подготовлен"}, 404)
            return self._send_media(media)
        if parsed.path == "/api/browse":
            requested = parse_qs(parsed.query).get("path", [str(Path.home())])[0]
            path = Path(requested).expanduser().resolve()
            if not path.is_dir():
                path = path.parent
            try:
                entries = []
                for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                    if item.name.startswith("."):
                        continue
                    entries.append({"name": item.name, "path": str(item), "directory": item.is_dir(),
                                    "suffix": item.suffix.lower()})
                return self._json({"path": str(path), "parent": str(path.parent), "entries": entries[:500]})
            except PermissionError:
                return self._json({"error": "Нет доступа к папке"}, 403)
        if parsed.path.startswith("/api/result"):
            with LOCK:
                output = STATE.get("output")
            if not output or not Path(output).is_file():
                return self._json({"error": "Результат не найден"}, 404)
            data = Path(output).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(output)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data); return
        if parsed.path == "/api/projection-result":
            with LOCK:
                variants = PROJECTION_STATE.get("variants", [])
            query = parse_qs(parsed.query)
            variant = int(query.get("variant", ["0"])[0])
            result = variants[variant] if 0 <= variant < len(variants) else None
            if not result:
                return self._json({"error": "Проекция не найдена"}, 404)
            return self._send_result_png(result, query.get("saved", ["0"])[0] == "1")
        if parsed.path == "/api/export-geotiff":
            # Generated on request next to the PNG, then streamed as a download.
            from .georaster import export_geotiff
            query = parse_qs(parsed.query)
            result = export_geotiff(query.get("path", [""])[0], query.get("crs", ["utm"])[0])
            data = Path(result["path"]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/tiff")
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(Path(result['path']).name)}")
            self.send_header("X-DroneVision-Path", quote(result["path"]))
            self.send_header("X-DroneVision-CRS", quote(result["crs"]))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data); return
        if parsed.path == "/api/raster":
            # Any raster written by save_georaster (sidecar required), e.g. results of «Улучшить».
            from .georaster import load_georaster
            query = parse_qs(parsed.query)
            meta = load_georaster(query.get("path", [""])[0])
            return self._send_result_png({"path": meta["image"], "data_path": meta["data"],
                                          "adjustments": meta.get("adjustments")},
                                         query.get("saved", ["0"])[0] == "1")
        if parsed.path == "/api/wave-progress":
            with LOCK:
                return self._json(dict(WAVE_STATE))
        if parsed.path == "/api/wave-result":
            # A result of «Течения и спектр волн» saved earlier (restoring a session).
            source = Path(parse_qs(parsed.query).get("path", [""])[0]).expanduser().resolve()
            if source.suffix.lower() != ".json" or "_currents" not in source.name or not source.is_file():
                return self._json({"error": "Результат расчёта течений не найден"}, 404)
            result = json.loads(source.read_text(encoding="utf-8"))
            if result.get("kind") != "wave_currents":
                return self._json({"error": "Это не результат расчёта течений"}, 400)
            return self._json(result)
        if parsed.path == "/api/mosaic-result":
            with LOCK:
                result = MOSAIC_STATE.get("result")
            if not result:
                return self._json({"error": "Мозаика не найдена"}, 404)
            query = parse_qs(parsed.query)
            return self._send_result_png(result, query.get("saved", ["0"])[0] == "1")
        relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR not in target.parents and target != STATIC_DIR:
            return self.send_error(404)
        if not target.is_file():
            return self.send_error(404)
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", STATIC_TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0]
                         or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        try:
            payload = self._payload()
            if self.path == "/api/inspect":
                config = ProjectConfig.from_dict(payload)
                meta = probe(config.video_path)
                return self._json({"video": meta, "config": asdict(config)})
            if self.path == "/api/viewer":
                from .viewer import viewer_data
                return self._json(viewer_data(payload["video_path"]))
            if self.path == "/api/proxy":
                from .viewer import ensure_proxy
                path = ensure_proxy(payload["video_path"], int(payload.get("height", 720)))
                return self._json({"path": str(path), "quality": f"{int(payload.get('height', 720))}p"})
            if self.path == "/api/project-frame":
                from .projection import project_selected_frame
                variant = project_selected_frame(
                    payload["video_path"], payload["time_s"], payload.get("project_dir", ""),
                    float(payload.get("takeoff_height_m") or 0.0),
                    float(payload.get("horizontal_fov_deg") or 0.0),
                    float(payload.get("resolution_m_per_px") or 0.0),   # 0 = auto, from the frame
                    float(payload.get("horizon_offset_percent") or 0.0),
                    float(payload.get("yaw_offset_deg") or 0.0),
                    adjustments=payload.get("adjustments"),
                    bottom_crop_percent=float(payload.get("bottom_crop_percent") or 0.0),
                    max_distance_m=float(payload.get("max_distance_m") or 10_000.0),
                )
                result = {"variants": [variant], "time_s": variant["time_s"], "frame_index": variant["frame_index"]}
                with LOCK:
                    PROJECTION_STATE["variants"] = [variant]
                return self._json(result)
            if self.path == "/api/project-frame-pair":
                from .projection import project_frame_pair
                result = project_frame_pair(
                    payload["video_path"], payload["frames"], payload.get("project_dir", ""),
                    float(payload.get("takeoff_height_m") or 0.0),
                    float(payload.get("horizontal_fov_deg") or 0.0),
                    float(payload.get("resolution_m_per_px") or 0.0),   # 0 = auto, from the frame
                    float(payload.get("yaw_offset_deg") or 0.0),
                    refine=bool(payload.get("refine_alignment", False)),
                    max_distance_m=float(payload.get("max_distance_m") or 10_000.0),
                )
                with LOCK:
                    PROJECTION_STATE["variants"] = result["variants"]
                return self._json(result)
            if self.path == "/api/mosaic-frames":
                from .mosaic import mosaic_selected_frames
                result = mosaic_selected_frames(
                    payload["video_path"], payload["times_s"], payload.get("project_dir", ""),
                    float(payload.get("takeoff_height_m") or 0.0),
                    float(payload.get("horizontal_fov_deg") or 0.0),
                    float(payload.get("resolution_m_per_px") or 0.0),   # 0 = auto, from the frames
                    float(payload.get("yaw_offset_deg") or 0.0),
                    float(payload.get("max_distance_m") or 3000.0),
                    adjustments=payload.get("adjustments"),
                    bottom_crop_percent=float(payload.get("bottom_crop_percent") or 0.0),
                    refine=bool(payload.get("refine_alignment", False)),
                )
                with LOCK:
                    MOSAIC_STATE["result"] = result
                return self._json(result)
            if self.path == "/api/average-interval":
                from .mosaic import average_interval
                result = average_interval(
                    payload["video_path"], float(payload["start_s"]), float(payload["end_s"]),
                    payload.get("project_dir", ""), float(payload.get("takeoff_height_m") or 0.0),
                    float(payload.get("horizontal_fov_deg") or 0.0),
                    float(payload.get("resolution_m_per_px") or 0.0),   # 0 = auto, from the frames
                    float(payload.get("yaw_offset_deg") or 0.0),
                    float(payload.get("max_distance_m") or 3000.0),
                    adjustments=payload.get("adjustments"),
                    bottom_crop_percent=float(payload.get("bottom_crop_percent") or 0.0),
                    refine=bool(payload.get("refine_alignment", False)),
                )
                with LOCK:
                    MOSAIC_STATE["result"] = result
                return self._json(result)
            if self.path == "/api/compare-intervals":
                # Both rasters are served by /api/raster, so the single-result MOSAIC_STATE is not touched.
                from .interval_compare import compare_averaged_intervals
                return self._json(compare_averaged_intervals(
                    payload["video_path"], payload["intervals"], payload.get("project_dir", ""),
                    float(payload.get("takeoff_height_m") or 0.0),
                    float(payload.get("horizontal_fov_deg") or 0.0),
                    float(payload.get("resolution_m_per_px") or 0.0),   # 0 = auto, from the frames
                    float(payload.get("yaw_offset_deg") or 0.0),
                    float(payload.get("max_distance_m") or 3000.0),
                    adjustments=payload.get("adjustments"),
                    bottom_crop_percent=float(payload.get("bottom_crop_percent") or 0.0),
                    refine=bool(payload.get("refine_alignment", False)),
                ))
            if self.path == "/api/enhance":
                from .enhance import enhance_region
                result = enhance_region(list(payload["paths"]), payload["region"], payload.get("adjustments"),
                                        payload.get("resolution_m_per_px"))
                return self._json(result)
            if self.path == "/api/profile":
                from .profiles import sample_profile, samples_json
                _, samples = sample_profile(payload["path"], payload["points"])
                return self._json(samples_json(samples))
            if self.path == "/api/curve-displacement":
                from .motion import estimate_curve_displacement
                return self._json(estimate_curve_displacement(
                    payload["points_a"], payload["points_b"], float(payload["delta_time_s"])))
            if self.path == "/api/track-contours":
                from .contour_tracking import track_contours
                return self._json(track_contours(payload["path_a"], payload["path_b"], list(payload["shapes"]),
                                                 float(payload.get("max_speed_m_s") or 1.5)))
            if self.path == "/api/motion-export":
                from .contour_tracking import export_motion
                return self._json(export_motion(payload["path_a"], payload["path_b"], list(payload["items"])))
            if self.path == "/api/profile-export":
                from .profiles import export_profile
                return self._json(export_profile(payload["path"], payload["points"], str(payload.get("name") or "line"),
                                                 str(payload.get("type") or "polyline"), str(payload.get("color") or "")))
            if self.path == "/api/calibrate-heading":
                from .calibration import estimate_heading
                project_dir = payload.get("project_dir") or ""
                return self._json(estimate_heading(
                    payload["video_path"], float(payload["time_s"]), float(payload.get("takeoff_height_m") or 0.0),
                    float(payload.get("horizontal_fov_deg") or 0.0),
                    output_dir=str(Path(project_dir).expanduser() / "calibration") if project_dir else None))
            if self.path == "/api/coast-view":
                from .coastview import coast_view
                return self._json(coast_view(
                    payload["video_path"], float(payload.get("time_s") or 0.0), float(payload.get("yaw_offset_deg") or 0.0),
                    float(payload.get("takeoff_height_m") or 0.0), float(payload.get("horizontal_fov_deg") or 0.0)))
            if self.path == "/api/coast-align":
                from .coastview import align
                return self._json(align(
                    payload["video_path"], float(payload.get("time_s") or 0.0), float(payload.get("yaw_offset_deg") or 0.0),
                    float(payload.get("takeoff_height_m") or 0.0), float(payload.get("horizontal_fov_deg") or 0.0)))
            if self.path == "/api/pose-offsets":
                from .coastview import save_alignment
                extra = {key: payload[key] for key in ("method", "measured_at_s", "yaw_offset_deg") if key in payload}
                return self._json(save_alignment(payload["video_path"], float(payload.get("pitch_offset_deg") or 0.0),
                                                 float(payload.get("roll_offset_deg") or 0.0), **extra))
            if self.path == "/api/wave-currents":
                from .wavefield import analyse
                with LOCK:
                    if WAVE_STATE["running"]:
                        return self._json({"error": "Расчёт течений уже выполняется"}, 409)
                    WAVE_STATE.update(running=True, progress=0.0, message="Подготовка")

                def progress(share, text):
                    with LOCK:
                        WAVE_STATE.update(progress=float(share), message=text)
                try:
                    start, end = payload.get("start_s"), payload.get("end_s")
                    return self._json(analyse(
                        payload["video_path"], None if start is None else float(start), None if end is None else float(end),
                        payload.get("project_dir", ""), float(payload.get("yaw_offset_deg") or 0.0),
                        float(payload.get("takeoff_height_m") or 0.0), float(payload.get("horizontal_fov_deg") or 0.0),
                        progress))
                finally:
                    with LOCK:
                        WAVE_STATE.update(running=False)
            if self.path == "/api/dialog":
                from .dialog import choose_path
                return self._json({"path": choose_path(payload.get("kind", "file"), payload.get("filter"),
                                                       payload.get("start"), payload.get("name"))})
            if self.path == "/api/data-root":
                from .catalog import choose_data_root
                return self._json(choose_data_root(payload["path"]))
            if self.path == "/api/session/save":
                from .session import save_session
                return self._json(save_session(payload.get("session"), payload.get("path"),
                                               bool(payload.get("keep_results"))))
            if self.path == "/api/session/load":
                from .session import load_session
                return self._json(load_session(payload.get("path")))
            if self.path == "/api/session/clear":
                from .session import clear_last_session
                return self._json(clear_last_session())
            if self.path == "/api/match":
                from .discovery import match_log
                return self._json(match_log(payload["video_path"], payload.get("log_directory")))
            if self.path == "/api/save":
                config = ProjectConfig.from_dict(payload)
                return self._json({"path": str(config.save())})
            if self.path == "/api/run":
                with LOCK:
                    if STATE["status"] == "running":
                        return self._json({"error": "Обработка уже выполняется"}, 409)
                    STATE.update(status="running", progress=0.0, message="Подготовка", output=None, artifacts={}, error=None)
                config = ProjectConfig.from_dict(payload)
                threading.Thread(target=_run, args=(config,), daemon=True).start()
                return self._json({"started": True}, 202)
            return self._json({"error": "Неизвестный запрос"}, 404)
        except Exception as exc:
            return self._json({"error": str(exc)}, 400)


def _run(config):
    try:
        from .mosaic import run
        def progress(value, message):
            with LOCK:
                STATE.update(progress=value, message=message)
        output = run(config, progress)
        manifest = json.loads((output.parent / "run.json").read_text(encoding="utf-8"))
        with LOCK:
            STATE.update(status="done", progress=1.0, message="Мозаика готова", output=str(output), artifacts=manifest.get("artifacts", {}))
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            STATE.update(status="error", message="Обработка остановлена", error=str(exc))


def serve(host="127.0.0.1", port=8765, open_browser=False):
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        # Usually a second start: the first copy is still running in another window.
        print(f"Порт {port} занят ({exc}).\nЕсли DroneVision уже запущен, он открыт по адресу {url}")
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        return
    print(f"DroneVision: {url}\nЭто окно должно оставаться открытым. Остановить: Ctrl+C или закрыть окно.")
    if open_browser:
        import webbrowser
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("DroneVision остановлен")
