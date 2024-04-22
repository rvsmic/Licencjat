import sys
import numpy as np
import matplotlib.pyplot as plt

# Funkcja do obliczania średniego czasu z pliku
def calculate_avg_time(filename):
    with open(filename, 'r') as file:
        times = [float(line.strip()) for line in file]
        return np.mean(times)
    
if len(sys.argv) != 3:
    print("Usage: python plot_diagram.py <filename> <num_samples>")
    sys.exit(1)

filename = sys.argv[1]
num_samples = int(sys.argv[2])

# Pliki z wynikami czasowymi
files = ['../DATA/OUT/base.time', '../DATA/OUT/cuda.time', '../DATA/OUT/sycl_cpu.time', '../DATA/OUT/sycl_gpu.time']

# Obliczanie średnich czasów
avg_times = [calculate_avg_time(filename) for filename in files]

# Wersje algorytmu
versions = ['C++ iteracyjnie', 'Cuda', 'Sycl CPU', 'Sycl GPU']

# Tworzenie wykresu
fig = plt.figure(figsize=(10, 6))
plt.bar(versions, avg_times, color=['blue', 'green', 'red', 'orange'])
plt.title('Porównanie średnich czasów wykonania algorytmu dla pliku ' + filename + ' (' + str(num_samples) + ' próbek czasowych)')
plt.xlabel('Wersje algorytmu')
plt.ylabel('Średni czas wykonania (ms)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Padding dla etykiet
label_padding = fig.get_figheight() * fig.dpi * 0.01

# Dodanie etykiet z wartościami nad słupkami
for i, time in enumerate(avg_times):
    plt.text(i, time + label_padding, f'{time:.2f} ms', ha='center')

# Zapisywanie wykresu do pliku
filename = filename.split('.')[0]
plt.savefig('../DOCS/histogram_' + filename + '.png')

# Wyświetlenie wykresu
plt.show()
