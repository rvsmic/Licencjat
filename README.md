# Praca Licencjacka
## Przenośna implementacja algorytmu cieniowania DEMów (ang. *Portable implementation of the DEM shading algorithm*)
**Autor:** Michał Rusinek
**Promotor:** dr hab. Przemysław Stpiczyński
**Uczelnia:** Uniwersytet Marii Curie-Skłodowskiej w Lublinie
**Kierunek:** Informatyka

### Opis repozytorium
Repozytorium podzielone zbudowane jest na ścisłej strukturze folderów:

- `DATA` - folder z danymi:
    - `IN` - folder z danymi wejściowymi
    - `OUT` - folder z danymi wyjściowymi
- `DOCS` - folder z dokumentacją
- `EXEC` - folder z plikami wykonywalnymi:
    - `HOME` - folder ze skryptami na komputer domowy
    - `LUNAR` - folder ze skryptami na klaster obliczeniowy UMCS Lunar
- `C++` - folder z implementacją iteracyjną w standardzie C++17
- `CUDA` - folder z implementacją równoległą w standardzie C++17 z użyciem CUDA
- `SYCL` - folder z implementacją równoległą w standardzie C++17 z użyciem SYCL
- `PYTHON` - folder zawierający testową (nieporuszoną w pracy i słabo zoptymalizowaną) implementację w języku Python oraz skrypt do generowania wykresów

### Wymagania
Do uruchomienia poszczególnych części repozytorium wymagane jest różne oprogramowanie (opisane wersje oprogramowania są sprawdzone, ale nie jest wykluczone, że programy będą działać z innymi wersjami):

- Implementacja w standardzie C++17:
    - Kompilator wspierający standard C++17
- Implementacja w standardzie C++17 z użyciem CUDA:
    - Kompilator wspierający standard C++17
    - NVIDIA CUDA Toolkit w wersji 11.7
- Implementacja w standardzie C++17 z użyciem SYCL:
    - Kompilator wspierający standard C++17
    - Dostęp do środowiska SYCL (np. Intel OneAPI) w wersji 2024.0.2
- Implementacja w języku Python i skrypt do generowania wykresów:
    - Interpreter Pythona w wersji 3.11.7
    - Biblioteki: numpy, matplotlib, PIL


### Instrukcja obsługi

Dane do przetworzenia powinny być umieszczone w folderze `DATA/IN` w formacie `*.tif`. Następnie należy wejśc do folderu `EXEC` i wybrać architekturę na jakiej ma być uruchomiony program:

- `HOME` to folder ze skryptami do uruchomienia programu na komputerze domowym
- `LUNAR` to folder ze skryptami do uruchomienia programu na klastrze obliczeniowym UMCS Lunar

Po wejściu w folder wybranej architektury należy uruchomić skrypt kompilacyjny (przy SYCL trzeba najpierw załadować zmienne środowiskowe skryptem `prepare_sycl_env.sh`), a następnie skrypt uruchamiający program. Wyniki zostaną zapisane w folderze `DATA/OUT` w formie podglądu JPEG i danych wyjściowych TIFF.