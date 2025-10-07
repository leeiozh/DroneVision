
# DroneVision

Набор скриптов для обработки видео и логов дрона DJI Mavic 3 из экспедиций Плавучего университета

Сейчас модуль может:
1) Выполнить проекцию и геопривязку видео, снятого под углом к горизонту (projection)
2) Восстановить поле скоростей объектов по видео, снятому в надир (tracking)



## Установка

Установка библиотек производится из файла requirements.txt, например, вот так для Windows

```bash
  py -m venv env_DV
  env_DV\Scripts\activate
  pip install -r requirements.txt
```

или вот так для Linux/macOS

```bash
  python3 -m venv env_DV
  source env_DV/bin/activate
  pip install -r requirements.txt
```

    
## Запуск

Для запсука в соответсвующем `config_*.py` необходимо указать путь к видео и логам. 

Чтобы преобразовать бинарные файлы логов в json-файл необходимо выполнить команду
```bash
./dji-log -o "output_file_name.json" "input_file_name"
```

Либо воспользоваться скриптом `convert_logs.sh`, адаптировав пути
