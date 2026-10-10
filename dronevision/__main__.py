import argparse
import json


def main():
    parser = argparse.ArgumentParser(prog="dronevision")
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser("web", help="Запустить локальный web-интерфейс")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--open", action="store_true", help="открыть страницу в браузере")
    setup = sub.add_parser("setup", help="Скачать внешние программы (Windows) и проверить установку")
    setup.add_argument("--target", choices=["windows"], default=None,
                       help="только скачать программы для Windows в bin/ (чтобы передать готовую папку)")
    setup.add_argument("--force", action="store_true", help="скачать заново")
    bundle = sub.add_parser("bundle", help="Собрать zip для передачи на компьютер с Windows")
    bundle.add_argument("--out", default=None, help="путь к архиву, по умолчанию dist/DroneVision-windows.zip")
    run = sub.add_parser("mosaic", help="Собрать мозаику из project.json")
    run.add_argument("config")
    calibrate = sub.add_parser("calibrate", help="Поправка курса по береговой линии в выбранных кадрах")
    calibrate.add_argument("video")
    calibrate.add_argument("--time", type=float, action="append", required=True,
                           help="время кадра, с; укажите несколько кадров с берегом вблизи")
    calibrate.add_argument("--takeoff-height", type=float, default=0.0, help="высота точки взлёта над водой, м")
    calibrate.add_argument("--fov", type=float, default=0.0, help="горизонтальный FOV без зума, 0 = профиль камеры")
    calibrate.add_argument("--out", default=None, help="папка для контрольных наложений")
    sync = sub.add_parser("check-sync", help="Сдвиг лога, фокусное и скачки курса по движению изображения")
    sync.add_argument("video")
    sync.add_argument("--start", type=float, default=0.0)
    sync.add_argument("--end", type=float, default=None)
    attitude = sub.add_parser("check-attitude",
                              help="Сравнить повороты камеры по изображению с углами из лога (дрейф, синхронизация)")
    attitude.add_argument("video")
    attitude.add_argument("--start", type=float, default=0.0)
    attitude.add_argument("--end", type=float, default=None)
    attitude.add_argument("--step", type=float, default=0.5, help="шаг между сравниваемыми кадрами, с")
    attitude.add_argument("--series", action="store_true", help="вывести накопленный дрейф по времени")
    attitude.add_argument("--save", action="store_true",
                          help="сохранить найденный сдвиг лога в <видео>.sync.json (учитывается всеми режимами)")
    coast = sub.add_parser("coastline", help="Береговая линия OSM: скачать (один раз, ~1 ГБ) и нарезать районы съёмок")
    coast.add_argument("--no-download", action="store_true", help="не скачивать, только нарезать из уже скачанного")
    coast.add_argument("--at", type=float, nargs=2, metavar=("LAT", "LON"), action="append",
                       help="дополнительная точка, для окрестностей которой нужна вырезка")
    args = parser.parse_args()
    from .paths import utf8_console
    utf8_console()
    if args.command == "web":
        from .web import serve
        serve(args.host, args.port, args.open)
    elif args.command == "setup":
        from .setup_tools import run as setup_run
        raise SystemExit(setup_run(args.target, args.force))
    elif args.command == "bundle":
        from .setup_tools import bundle
        bundle(args.out)
    elif args.command == "mosaic":
        from .config import ProjectConfig
        from .mosaic import run
        print(run(ProjectConfig.load(args.config), lambda p, m: print(f"{p:6.1%} {m}")))
    elif args.command == "coastline":
        from . import coastdata
        if coastdata.planet_file() is None and not args.no_download:
            print("Скачиваю полигоны суши OpenStreetMap (один раз)…")
            coastdata.download_planet()
        if coastdata.planet_file() is None:
            raise SystemExit("Файл полигонов суши не найден: запустите команду без --no-download")
        made = coastdata.prepare_for_catalogue()
        for lat, lon in args.at or []:
            path = coastdata.land_file_around(lat, lon)
            print(f"  {lat:.3f}, {lon:.3f}: {path}")
        print(json.dumps(coastdata.status(), ensure_ascii=False, indent=2))
        print(f"Готово: новых вырезок {len(made)}.")
    elif args.command == "calibrate":
        from .calibration import estimate_headings
        print(json.dumps(estimate_headings(args.video, args.time, args.takeoff_height, args.fov, args.out),
                         ensure_ascii=False, indent=2))
    elif args.command == "check-attitude":
        from .calibration import check_attitude
        from pathlib import Path
        from .discovery import sync_sidecar
        sidecar = sync_sidecar(Path(args.video).expanduser().resolve())
        previous = json.loads(sidecar.read_text(encoding="utf-8")).get("extra_log_shift_s", 0.0) if sidecar.is_file() else 0.0
        result = check_attitude(args.video, args.start, args.end, args.step)
        if args.save:
            # The measurement is relative to the sync already in use, which includes any earlier sidecar.
            total = round(previous + result["best_extra_log_shift_s"], 3)
            sidecar.write_text(json.dumps({"extra_log_shift_s": total, "measured_on_s": [args.start, args.end],
                                           "method": "check-attitude"}, indent=2), encoding="utf-8")
            result["saved"] = {"path": str(sidecar), "extra_log_shift_s": total}
        if not args.series:
            result.pop("drift_series")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        from .calibration import check_sync
        print(json.dumps(check_sync(args.video, args.start, args.end), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
