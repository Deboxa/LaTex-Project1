import numpy as np
from mpi4py import MPI
import csv
import time
from scipy.signal import convolve2d

# Параметры симуляции
N_SIMULATIONS = 1000     # Увеличено для статистической значимости
GRID_SIZE = 20              # Уменьшено для быстрого формирования структур
MAX_GENERATIONS = 400       # Оптимальное количество поколений
P_VALUES = [0.1, 0.2, 0.3] # Проверяемые плотности
KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.uint8)

def initialize_grid(p):
    """Инициализация сетки с биномиальным распределением"""
    return np.random.binomial(1, p, (GRID_SIZE, GRID_SIZE))

def count_neighbors(grid):
    """Подсчет соседей с тороидальными границами"""
    return convolve2d(grid, KERNEL, mode='same', boundary='wrap')

def update_grid(grid):
    """Обновление сетки по правилам игры Жизнь"""
    neighbors = count_neighbors(grid)
    new_grid = np.zeros_like(grid)
    new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    new_grid[(grid == 0) & (neighbors == 3)] = 1
    return new_grid

def simulate(p, rank):
    """Запуск одной симуляции с отслеживанием статичных структур"""
    np.random.seed((int(time.time()) + rank) % 2**32)
    grid = initialize_grid(p)
    prev_grid = grid.copy()
    stability = np.zeros(MAX_GENERATIONS, dtype=int)
    
    for gen in range(MAX_GENERATIONS):
        new_grid = update_grid(grid)
        
        # Проверка на статическую стабильность
        if np.array_equal(new_grid, grid):
            stability[gen] = 1
            break  # Прекращаем симуляцию после обнаружения
            
        prev_grid = grid.copy()
        grid = new_grid.copy()
    
    return stability

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for p in P_VALUES:
        # Распределение задач между процессами
        sims_per_proc = N_SIMULATIONS // size
        extra = N_SIMULATIONS % size
        local_sims = sims_per_proc + (1 if rank < extra else 0)

        # Локальные вычисления
        local_counts = np.zeros(MAX_GENERATIONS, dtype=int)
        for _ in range(local_sims):
            local_counts += simulate(p, rank)

        # Сбор результатов
        global_counts = np.zeros(MAX_GENERATIONS, dtype=int)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)

        # Сохранение данных (только root-процесс)
        if rank == 0:
            probabilities = global_counts / N_SIMULATIONS
            
            filename = f"static_stability_p_{p:.1f}.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Generation', 'Probability'])
                for gen, prob in enumerate(probabilities):
                    writer.writerow([gen+1, f"{prob:.6f}"])

    if rank == 0:
        print("Симуляция завершена. Результаты сохранены в CSV файлы.")