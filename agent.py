from collections import deque
import heapq
import math
import random


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:

    def sense_and_act(self, percept):
        if percept['wall_ahead']:
            return random.choice(['Left', 'Right', 'Down'])

        return 'Up'


class ModelBasedAgent:

    def __init__(self):
        self.last_action = None

    def sense_and_act(self, percept):
        if percept['wall_ahead']:
            actions = ['Left', 'Right', 'Down']

            if self.last_action in actions:
                index = actions.index(self.last_action)
                action = actions[(index + 1) % len(actions)]
            else:
                action = actions[0]
        else:
            action = 'Up'

        self.last_action = action
        return action


class SearchAgent:

    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar'
        self.moves = [
            ('Up', (0, 1)),
            ('Right', (1, 0)),
            ('Down', (0, -1)),
            ('Left', (-1, 0))
        ]

    def get_successors(self, state, walls, grid_size):
        width, height = grid_size

        for action, (dx, dy) in self.moves:
            next_state = (state[0] + dx, state[1] + dy)

            if (
                0 <= next_state[0] < width
                and 0 <= next_state[1] < height
                and next_state not in walls
            ):
                yield next_state, action

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt(
            (pos[0] - goal[0]) ** 2
            + (pos[1] - goal[1]) ** 2
        )

    def bfs_search(self, start, goal, walls, grid_size):
        walls = set(walls)
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            state, path = frontier.popleft()

            if state == goal:
                return path

            for next_state, action in self.get_successors(
                state,
                walls,
                grid_size
            ):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))

        return None

    def dfs_search(self, start, goal, walls, grid_size):
        walls = set(walls)
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            state, path = frontier.pop()

            if state == goal:
                return path

            successors = list(self.get_successors(state, walls, grid_size))

            for next_state, action in reversed(successors):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))

        return None

    def ucs_search(self, start, goal, walls, grid_size):
        walls = set(walls)
        frontier = [(0, start, [])]
        reached = set()

        while frontier:
            cost, state, path = heapq.heappop(frontier)

            if state in reached:
                continue

            reached.add(state)

            if state == goal:
                return path

            for next_state, action in self.get_successors(
                state,
                walls,
                grid_size
            ):
                if next_state not in reached:
                    heapq.heappush(
                        frontier,
                        (cost + 1, next_state, path + [action])
                    )

        return None

    def astar_search(
        self,
        start_pos,
        goal_pos,
        walls,
        grid_size,
        heuristic_type='manhattan'
    ):
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        walls = set(walls)

        if heuristic_type.lower() == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            heuristic = self.manhattan_distance

        start_h = heuristic(start_pos, goal_pos)
        frontier = []
        heapq.heappush(frontier, (start_h, 0, start_pos, []))
        reached_states = set()
        costs = {start_pos: 0}

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(
                frontier
            )

            if current_pos in reached_states:
                continue

            if current_pos == goal_pos:
                return path_taken

            reached_states.add(current_pos)

            for next_pos, action in self.get_successors(
                current_pos,
                walls,
                grid_size
            ):
                if next_pos in reached_states:
                    continue

                new_g = g_cost + 1

                if new_g < costs.get(next_pos, float('inf')):
                    costs[next_pos] = new_g
                    new_h = heuristic(next_pos, goal_pos)
                    new_f = new_g + new_h
                    heapq.heappush(
                        frontier,
                        (
                            new_f,
                            new_g,
                            next_pos,
                            path_taken + [action]
                        )
                    )

        return None

    def sense_and_act(self, percept):
        if not self.plan:
            start = tuple(percept['agent_pos'])
            walls = percept['walls']
            grid_size = percept['grid_size']
            foods = sorted(
                percept['all_food'],
                key=lambda food: abs(food[0] - start[0])
                + abs(food[1] - start[1])
            )
            if self.active_algo == 'BFS':
                search = self.bfs_search
            elif self.active_algo == 'DFS':
                search = self.dfs_search
            elif self.active_algo == 'UCS':
                search = self.ucs_search
            elif self.active_algo == 'AStar':
                search = self.astar_search
            else:
                search = self.bfs_search

            for goal in foods:
                path = search(start, goal, walls, grid_size)

                if path:
                    self.plan = path
                    break

        if self.plan:
            return self.plan.pop(0)

        return None
