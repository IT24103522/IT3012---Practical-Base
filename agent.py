from collections import deque
import heapq
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
        self.active_algo = 'BFS'
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
            searches = {
                'BFS': self.bfs_search,
                'DFS': self.dfs_search,
                'UCS': self.ucs_search
            }
            search = searches.get(self.active_algo.upper(), self.bfs_search)

            for goal in foods:
                path = search(start, goal, walls, grid_size)

                if path:
                    self.plan = path
                    break

        if self.plan:
            return self.plan.pop(0)

        return None
