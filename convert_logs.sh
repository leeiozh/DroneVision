#!/bin/bash

# Папка с файлами
folder="VideoMAVIC/FlightRecords"

# Перебор всех файлов в указанной папке
for file in "$folder"/*; do
    # Проверяем, что это файл
    if [[ -f "$file" ]]; then
        # Извлекаем дату и время из имени файла, например: DJIFlightRecord_2023-07-18_[03-41-06].txt
        filename=$(basename "$file")
        datetime=$(echo "$filename" | sed -E 's/DJIFlightRecord_([0-9]{4})-([0-9]{2})-([0-9]{2})_\[([0-9]{2})-([0-9]{2})-([0-9]{2})\].txt/\1\2\3T\4\5\6/')
        
        # Выполняем команду
        ./dji-log -o "VideoMavic/logs/$datetime.json" "$file"
    fi
done

