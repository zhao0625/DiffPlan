import numpy as np
from gym_miniworld.miniworld import MiniWorldEnv

from utils.vis_pano import plot_pano


class MazeNavWorld(MiniWorldEnv):
    def __init__(
            self,
            maze_map=None,
            obs_width=80,
            obs_height=60,
            domain_rand=False
    ):
        if maze_map is not None:
            self.num_rows, self.num_cols = maze_map.shape[0], maze_map.shape[1]
            self.maze_map = maze_map
            self.valid_pos = self.get_valid_pos(self.maze_map, valid_sym=1, return_np=False)

        self.room_size = 1
        self.gap_size = 0.01  # room gap size / wall thickness

        self.num_views = 4
        self.num_rgb = 3
        self.obs_dim = (obs_height, obs_width, self.num_rgb)

        self.obs_width = obs_width
        self.obs_height = obs_height

        super().__init__(obs_width=obs_width, obs_height=obs_height, domain_rand=domain_rand)

    def set_map(self, maze_map):
        self.num_rows, self.num_cols = maze_map.shape[0], maze_map.shape[1]
        self.maze_map = maze_map
        self.valid_pos = self.get_valid_pos(maze_map, valid_sym=1, return_np=False)

    def reset(self):
        # Note: `_gen_world()` is invoked
        return super().reset()

    def _gen_world(self):
        _rooms = {}

        # > Note: row is for x-axis, col is for z-axis, full position coordinate is like (x, -, z)
        for _row in range(self.num_rows):
            for _col in range(self.num_cols):
                # check if the current cell is a corridor cell
                if (_row, _col) not in self.valid_pos:
                    continue

                # compute the boundary
                min_x = _col * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = _row * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                # add the room
                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex='brick_wall'
                )

                _rooms[_row, _col] = room

        visited = set()

        # > connect the neighbors rooms given map
        for _row, _col in self.valid_pos:
            room = _rooms[_row, _col]
            visited.add(room)

            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, left, right

            # loop valid neighbors
            for d_row, d_col in neighbors:
                n_row = _row + d_row
                n_col = _col + d_col

                if n_row < 0 or n_row >= self.num_rows:
                    continue
                if n_col < 0 or n_col >= self.num_cols:
                    continue

                # > don't render rooms for invalid positions
                if (n_row, n_col) not in self.valid_pos:
                    continue

                neighbor = _rooms[n_row, n_col]

                if neighbor in visited:
                    continue

                if d_col == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif d_row == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

        # > no need to place goal
        # self.box = self.place_entity(Box(color='red'))

        self.place_agent()

    def render_pano(self, pos, vis_top=False):
        """
        Helper for rendering egocentric panoramic images
        """

        pano_obs = np.empty(
            shape=(self.num_views, self.obs_height, self.obs_width, self.num_rgb),
            dtype=np.uint8
        )

        # > set position; note: swap col and row outside!
        self.agent.pos = np.array([pos[0] + 0.5, 0., pos[1] + 0.5])

        # > (1) agent facing right, so (3/2)pi or (-1/2)pi should be north
        # > (2) should have counter-clockwise order
        directions = np.array([1. / 2, 0. / 2, 3. / 2, 2. / 2]) * np.pi
        # directions = np.arange(0, 2 * np.pi, np.pi / 2)
        # directions = np.roll(directions, -1)

        for _i, _dir in enumerate(directions):
            # > set direction/orientation
            self.agent.dir = _dir

            if vis_top:
                print(f'(Debug) position = {self.agent.pos}, orientation = {self.agent.dir}')

            # > retrieve observation
            pano_obs[_i] = self.render_obs()

        # > retrieve top-down observation for visualization
        if vis_top:
            top_obs = self.render_top_view()
            return pano_obs, top_obs
        else:
            return pano_obs

    def get_all_pano(self, vis_top=False, plot_top=False):
        # > panoramic (4-direction) RGB images (e.g., 32 x 32 x 3) for every location (e.g., 5 x 5)
        # > e.g. 5 x 5 x (4 x 32 x 32 x 3)
        pos2pano = np.zeros((self.num_rows, self.num_cols, self.num_views) + self.obs_dim,
                            dtype=np.uint8)
        pos2top = {}

        # > fill actual images
        for pos in self.valid_pos:
            # > note: swap col and row!
            pos_agent = (pos[1], pos[0])

            if vis_top:
                pano_obs, top_obs = self.render_pano(pos=pos_agent, vis_top=True)
                pos2top[pos_agent] = top_obs

                if plot_top:
                    plot_pano(pano_obs, top_obs, title=f'Position = ${pos}$')

            else:
                pano_obs = self.render_pano(pos=pos_agent, vis_top=False)

            pos2pano[pos[1]][pos[0]] = pano_obs

        return (pos2pano, pos2top) if vis_top else pos2pano

    @staticmethod
    def get_valid_pos(maze_map, valid_sym=0, return_np=False):
        """return valid (0, empty) grid cells for goal/start position"""
        res = np.argwhere(maze_map == valid_sym)
        return res if return_np else list(map(tuple, res.tolist()))
