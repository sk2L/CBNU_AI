import sys
import numpy as np
import heapq
import os
import warnings
import random
import logging

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# 로깅 설정 (파일 및 콘솔에 동시에 출력되도록 설정)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pathfinding.log', mode='w'),  # 로그를 파일에 저장
        logging.StreamHandler(sys.stdout)  # 로그를 콘솔에 출력
    ]
)


# 유클리드 거리 기반 휴리스틱 함수 (3D 공간)
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


# 3D 이웃 탐색 함수 (상하좌우 + 상하좌우 대각선 + 위아래 고도 고려)
def get_neighbors_3d(x, y, z, shape):
    neighbors = []
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                  (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
                  (0, 0, 1), (0, 0, -1)]

    for dx, dy, dz in directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
            neighbors.append((nx, ny, nz))

    return neighbors


# 고도 데이터를 npy 파일에서 불러오는 함수
def load_elevation_data_npy(file_path):
    logging.info(f"Loading elevation data from {file_path}")
    elevation_data = np.load(file_path)
    logging.info(f"Elevation data loaded successfully. Shape: {elevation_data.shape}")
    return elevation_data


# 장애물 데이터를 npy 파일에서 불러오는 함수
def load_obstacle_data(file_path):
    logging.info(f"Loading obstacle data from {file_path}")
    obstacle_data = np.load(file_path)
    logging.info(f"Obstacle data loaded successfully. Shape: {obstacle_data.shape}")
    return obstacle_data


# A* 알고리즘 (고도와 장애물 반영)
def a_star_algorithm_3d(start, goal, elevation_data, obstacle_data, elevation_weight=10):
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    logging.info(f"Started A* algorithm from {start} to {goal}")

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            logging.info(f"Reached goal: {goal}")
            break

        x, y, z = current
        for neighbor in get_neighbors_3d(x, y, z, elevation_data.shape):
            nx, ny, nz = neighbor

            if obstacle_data[nx, ny, nz] == 0:  # 장애물이 없는 곳만 탐색
                elevation_diff = abs(elevation_data[nx, ny, nz] - elevation_data[x, y, z])
                new_cost = cost_so_far[current] + np.sqrt(1 + (elevation_weight * elevation_diff) ** 2)
                priority = new_cost + heuristic(goal, neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(open_heap, (priority, neighbor))
                    came_from[neighbor] = current

    logging.info(f"Finished A* algorithm from {start} to {goal}")
    return reconstruct_path(came_from, start, goal)


# 경로를 재구성하는 함수
def reconstruct_path(came_from, start, current):
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


# 시작점과 목표점을 랜덤으로 생성하는 함수 (장애물이 없는 곳에서만 설정)
def generate_random_start_goal(obstacle_data, max_attempts=1000):
    shape = obstacle_data.shape
    attempts = 0
    while attempts < max_attempts:
        start = (random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1), random.randint(0, shape[2] - 1))
        goal = (random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1), random.randint(0, shape[2] - 1))
        if obstacle_data[start] == 0 and obstacle_data[goal] == 0 and start != goal:
            logging.info(f"Generated random start {start} and goal {goal}")
            return start, goal
        attempts += 1

    raise ValueError("Could not find valid start and goal positions after many attempts.")


# 개별 장애물에 대해 경로 생성 함수
def process_single_obstacle(i, obstacle_file, elevation_data, save_directory):
    try:
        logging.info(f"Processing obstacle file {i}: {obstacle_file}")
        obstacle_data = load_obstacle_data(obstacle_file)

        # 데이터 유효성 검사
        logging.info(f"Obstacle data shape: {obstacle_data.shape}, Elevation data shape: {elevation_data.shape}")
        if obstacle_data.shape != elevation_data.shape:
            raise ValueError(f"Mismatch in obstacle and elevation data shapes for file {i}")

        start, goal = generate_random_start_goal(obstacle_data)  # 랜덤으로 시작점과 목표점 생성

        logging.info(f"Start: {start}, Goal: {goal}")
        path = a_star_algorithm_3d(start, goal, elevation_data, obstacle_data, elevation_weight=10)
        if path:
            save_path_data(path, i, save_directory)
            logging.info(f"Pathfinding successful for obstacle file {i}")
        else:
            logging.error(f"Pathfinding failed for obstacle file {i}")
    except Exception as e:
        logging.error(f"Error processing obstacle file {i}: {e}")


# 최단 경로 탐색 결과를 파일로 저장하는 함수
def save_path_data(path, index, save_directory="path_data_final"):
    os.makedirs(save_directory, exist_ok=True)
    file_path = os.path.join(save_directory, f"path_data_{index}.npy")
    np.save(file_path, np.array(path))
    logging.info(f"Saved path data {index} to {file_path}")
    print(f"Saved path data {index} to {file_path}")  # 콘솔에 출력


# 이미 저장된 경로 파일들의 인덱스를 찾는 함수
def get_last_path_index(save_directory="path_data_final"):
    existing_files = os.listdir(save_directory)
    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith('path_data_')]
    if indices:
        return max(indices) + 1  # 가장 큰 인덱스에서 시작
    return 0  # 파일이 없으면 0부터 시작

# 경로 탐색을 순차적으로 처리하는 함수 (이미 생성된 파일 이후부터 새로 생성)
def generate_and_save_paths(obstacle_files, elevation_file, save_directory="path_data_final"):
    os.makedirs(save_directory, exist_ok=True)
    elevation_data = load_elevation_data_npy(elevation_file)

    # 이미 생성된 파일의 마지막 인덱스 확인
    start_index = get_last_path_index(save_directory)

    for i, obstacle_file in enumerate(obstacle_files[start_index:], start=start_index):
        logging.info(f"Processing file {i}: {obstacle_file}")
        process_single_obstacle(i, obstacle_file, elevation_data, save_directory)

# 메인 모듈 확인 (Windows에서 멀티프로세싱 오류 방지)
if __name__ == '__main__':
    logging.info("Started pathfinding process")

    # 경로 확인을 위한 로깅 추가
    obstacle_files = [f"C:/02.논문/pythonProject/Obstacle/ObstacleDataFinal/obstacle_data_3d_grid_{i}.npy" for i in
                      range(100)]
    elevation_file = "C:/02.논문/pythonProject/DTED/elevation_data_C5.npy"

    # 파일 경로를 출력하여 경로가 올바른지 확인
    for file in obstacle_files:
        if not os.path.exists(file):
            logging.error(f"File not found: {file}")

    if not os.path.exists(elevation_file):
        logging.error(f"Elevation file not found: {elevation_file}")

    # 경로 탐색을 순차적으로 실행
    generate_and_save_paths(obstacle_files, elevation_file)
    logging.info("Completed pathfinding process")
