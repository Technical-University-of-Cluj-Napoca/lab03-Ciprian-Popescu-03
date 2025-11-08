from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot
import math

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Breadth-First Search (BFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    queue = deque()
    queue.append(start)
    visited = {start}
    came_from = {}

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
        current = queue.popleft()

        if current == end:
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()
        
        draw()

        if current != start:
            current.make_closed()
    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depdth-First Search (DFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = stack.pop()

        if current == end:
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()
        
        draw()

        if current != start:
            current.make_closed()

    return False
                 


def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Manhattan distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Euclidian distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def astar(draw: callable, grid: Grid, start: Spot, end: Spot, heuristic: callable) -> bool:
    """
    A* Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    count = 0  # to break ties in PriorityQueue
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid.grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid.grid for spot in row}
    f_score[start] = heuristic(start.get_position(), end.get_position())

    open_set_hash = {start}  # to quickly check if a node is in the open set

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]  # get Spot from PriorityQueue
        open_set_hash.remove(current)

        if current == end:
            # reconstruct path
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            start.make_start()
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1  # assume cost = 1 for each move
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor.get_position(), end.get_position())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

MAX_DEPTH = 20  # example depth limit for DLS

def dls(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    stack = [(start, 0)]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        current, depth = stack.pop()

        if current == end:
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            return True
        
        if depth < MAX_DEPTH:
            for neighbor in current.neighbours:
                if neighbor not in visited and not neighbor.is_barrier():
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append((neighbor, depth + 1))
                    neighbor.make_open()
        
        draw()

        if current != start:
            current.make_closed()
    
    return False



def uniform_cost_search(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Uniform Cost Search (Uninformed search / Dijkstra) on a grid.
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid.grid for spot in row}
    g_score[start] = 0
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            # reconstruct path
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            start.make_start()
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            temp_g_score = g_score[current] + 1  # assume cost = 1 for each move
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False


def dijkstra(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Dijkstra's Algorithm (Uniform-Cost Search) on a grid.
    Expands nodes based on the lowest cost from start (g(n)).
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid.grid for spot in row}
    g_score[start] = 0
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            # reconstruct path
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()  # mark path
                    draw()
            start.make_start()
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue

            temp_g_score = g_score[current] + 1  # assume cost = 1 per move
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def depth_limited_search(draw: callable, current: Spot, end: Spot, depth: int, came_from: dict, visited: set) -> bool:
    """
    DFS up to a maximum depth.
    """
    if current == end:
        return True
    if depth <= 0:
        return False

    for neighbor in current.neighbors:
        if neighbor.is_barrier() or neighbor in visited:
            continue

        visited.add(neighbor)
        came_from[neighbor] = current
        neighbor.make_open()
        draw()
        if depth_limited_search(draw, neighbor, end, depth-1, came_from, visited):
            return True
        draw()
    if current.color != COLORS["ORANGE"]:
        current.make_closed()
        draw()
    return False

def iterative_deepening_search(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Iterative Deepening Search (IDS) on the grid.
    """
    max_depth = grid.rows * grid.cols  # maximum possible depth
    for depth in range(max_depth):
        visited = {start}
        came_from = {}
        if depth_limited_search(draw, start, end, depth, came_from, visited):
            # reconstruct path
            current = end
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            start.make_start()
            end.make_end()
            return True
    return False

def ida_star_draw_path(came_from: dict, start: Spot, end: Spot, draw: callable):
    current = end
    while current in came_from:
        current = came_from[current]
        if current != start:
            current.make_path()
            draw()
    start.make_start()
    end.make_end()


def ida_star_recursive(draw: callable, current: Spot, end: Spot, g: float, bound: float,
                       came_from: dict, heuristic: callable) -> tuple[bool, float]:
    """
    Recursive DFS-like search with cost bound.
    Returns: (found, next_bound)
    """
    f = g + heuristic(current.get_position(), end.get_position())
    if f > bound:
        return False, f
    if current == end:
        return True, f

    min_bound = float('inf')
    for neighbor in current.neighbors:
        if neighbor.is_barrier() or neighbor in came_from:
            continue
        came_from[neighbor] = current
        neighbor.make_open()
        found, temp_bound = ida_star_recursive(draw, neighbor, end, g + 1, bound, came_from, heuristic)
        if found:
            return True, temp_bound
        if temp_bound < min_bound:
            min_bound = temp_bound
        draw()
    if current.color != COLORS["ORANGE"]:
        current.make_closed()
    return False, min_bound


def ida_star(draw: callable, grid: Grid, start: Spot, end: Spot, heuristic: callable) -> bool:
    """
    Iterative Deepening A* (IDA*) Algorithm.
    """
    bound = heuristic(start.get_position(), end.get_position())
    came_from = {}
    while True:
        found, temp_bound = ida_star_recursive(draw, start, end, 0, bound, came_from, heuristic)
        if found:
            ida_star_draw_path(came_from, start, end, draw)
            return True
        if temp_bound == float('inf'):
            return False  
        bound = temp_bound



# and the others algorithms...
# ▢ Depth-Limited Search (DLS)
# ▢ Uninformed Cost Search (UCS)
# ▢ Greedy Search
# ▢ Iterative Deepening Search/Iterative Deepening Depth-First Search (IDS/IDDFS)
# ▢ Iterative Deepening A* (IDA)
# Assume that each edge (graph weight) equalss