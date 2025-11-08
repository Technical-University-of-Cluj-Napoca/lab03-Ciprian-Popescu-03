from utils import *
from grid import Grid
from searching_algorithms import *

if __name__ == "__main__":
    print("Path Visualizing Algorithm Controls:")
    print("LEFT CLICK: Set start / end / barriers")
    print("RIGHT CLICK: Remove start / end / barriers")
    print("B: BFS (Breadth-First Search)")
    print("D: DFS (Depth-First Search)")
    print("L: DLS (Depth-Limited Search)")
    print("A: A* Search (Press M for Manhattan, E for Euclidean)")
    print("U: UCS (Uniform Cost Search)")
    print("G: Dijkstra / Greedy Search")
    print("I: IDS (Iterative Deepening Search)")
    print("Q: IDA* Search (Press M for Manhattan, E for Euclidean)")
    print("C: Clear the grid")
    print("ESC / Close window: Quit")
    
    pygame.init()

    # setting up how big will be the display window
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))

    # set a caption for the window
    pygame.display.set_caption("Path Visualizing Algorithm")

    ROWS = 50  # number of rows
    COLS = 50  # number of columns
    grid = Grid(WIN, ROWS, COLS, WIDTH, HEIGHT)

    start = None
    end = None

    # flags for running the main loop
    run = True
    started = False

    while run:
        grid.draw()  
        for event in pygame.event.get():
            # verify what events happened
            if event.type == pygame.QUIT:
                run = False

            if started:
                # do not allow any other interaction if the algorithm has started
                continue  # ignore other events if algorithm started

            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                pos = pygame.mouse.get_pos()
                row, col = grid.get_clicked_pos(pos)

                if row >= ROWS or row < 0 or col >= COLS or col < 0:
                    continue  # ignore clicks outside the grid

                spot = grid.grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT CLICK
                pos = pygame.mouse.get_pos()
                row, col = grid.get_clicked_pos(pos)
                spot = grid.grid[row][col]
                spot.reset()

                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                # Run BFS Algorithm
                if event.key == pygame.K_b and not started:
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)
                    bfs(lambda: grid.draw(), grid, start, end)
                    started = False
                
                # Run DFS Algorithm
                if event.key == pygame.K_d and not started:
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)
                    dfs(lambda: grid.draw(), grid, start, end)
                    started = False

                # Run DLS Algorithm
                if event.key == pygame.K_l and not started:
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)
                    dfs(lambda: grid.draw(), grid, start, end)
                    started = False

                # Run A* Algorithm
                if event.key == pygame.K_a and not started:  
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)

                    # Choose heuristic based on another key
                    print("Press M for Manhattan, E for Euclidean")
                    waiting = True
                    while waiting:
                        for e in pygame.event.get():
                            if e.type == pygame.KEYDOWN:
                                if e.key == pygame.K_m:
                                    heuristic_fn = h_manhattan_distance
                                    waiting = False
                                elif e.key == pygame.K_e:
                                    heuristic_fn = h_euclidian_distance
                                    waiting = False

                    astar(lambda: grid.draw(), grid, start, end, heuristic=heuristic_fn)
                    started = False

                if event.key == pygame.K_u and not started:  # U for UCS
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)
                    uniform_cost_search(lambda: grid.draw(), grid, start, end)
                    started = False

                # Run Dijkstra (UCS)
                if event.key == pygame.K_g and not started:  # G for Greedy / Dijkstra
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)

                    dijkstra(lambda: grid.draw(), grid, start, end)
                    started = False

                # Run IDS Algorithm
                if event.key == pygame.K_i and not started: 
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)

                    iterative_deepening_search(lambda: grid.draw(), grid, start, end)
                    started = False

                # Run IDA* Algorithm
                if event.key == pygame.K_q and not started:  # A for IDA*
                    started = True
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)
                    
                    # choose heuristic
                    print("Press M for Manhattan, E for Euclidean")
                    waiting = True
                    while waiting:
                        for e in pygame.event.get():
                            if e.type == pygame.KEYDOWN:
                                if e.key == pygame.K_m:
                                    heuristic_fn = h_manhattan_distance
                                    waiting = False
                                elif e.key == pygame.K_e:
                                    heuristic_fn = h_euclidian_distance
                                    waiting = False
                    
                    ida_star(lambda: grid.draw(), grid, start, end, heuristic=heuristic_fn)
                    started = False


                # Clear the grid
                if event.key == pygame.K_c:
                    print("Clearing the grid...")
                    start = None
                    end = None
                    grid.reset()
                
                if event.key == pygame.K_ESCAPE:
                    run = False
    pygame.quit()
