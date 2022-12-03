import numpy as np
from physical_objects import Isolated, Stick, Hinge, Complex
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")


class System:
    def __init__(self, n_isolated, n_stick, n_hinge, n_complex=0, complex_sizes=None, delta_t=0.001,
                 box_size=None, loc_std=1., vel_norm=0.5,
                 interaction_strength=1., charge_types=None,
                 ):
        self.n_isolated, n_stick, n_hinge = n_isolated, n_stick, n_hinge
        self.n_complex = n_complex
        self.complex_sizes = complex_sizes
        self.delta_t = delta_t
        self._max_F = 0.1 / self.delta_t  # tentative setting
        self.box_size = box_size
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.dim = 3

        if self.n_complex > 0:
            if complex_sizes is None:
                raise NotImplementedError('Really want complex bodies with random sizes?')
            rand_size = complex_sizes
            assert len(complex_sizes) == n_complex
            offset = np.sum(rand_size)
        else:
            rand_size = None
            offset = 0

        self.n_balls = n_isolated * 1 + n_stick * 2 + n_hinge * 3

        if self.n_complex > 0:
            self.n_balls += offset

        n = self.n_balls
        # self.loc_std = loc_std * (float(self.n_balls) / 5.) ** (1 / 3)
        self.loc_std = loc_std * (float(self.n_balls) / 5.) ** (1 / 3) + 0.5

        if charge_types is None:
            charge_types = [1.0, -1.0]
        self.charge_types = charge_types

        self.diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(self.diag_mask, 0)

        charges = np.random.choice(self.charge_types, size=(self.n_balls, 1))
        self.charges = charges
        edges = charges.dot(charges.transpose())
        self.edges = edges

        assert self.n_isolated == 0
        # Hyper-parameters
        eps = 1.6  # the minimum distance between centers
        max_try = 20
        X_c = np.zeros((self.n_complex, self.dim))
        for i in range(X_c.shape[0]):
            # sample
            counter = 0
            while True:
                xx = 2 * (np.random.rand(self.dim) - 0.5) * self.loc_std
                ok = True
                for j in range(i):
                    d = np.sqrt(np.sum((X_c[j] - xx) ** 2, axis=-1))
                    if d < eps:
                        ok = False
                        break
                if ok:
                    X_c[i] = xx
                    break
                else:
                    counter += 1
                    if counter >= max_try:
                        # have to accept
                        print('max try, have to accept')
                        X_c[i] = xx
                        break
        min_d = 1e10
        for i in range(X_c.shape[0]):
            for j in range(i + 1, X_c.shape[0]):
                dd = np.sqrt(np.sum((X_c[i] - X_c[j]) ** 2, axis=-1))
                min_d = min(min_d, dd)
        min_d = min_d / 1.6
        # print(min_d)
        # Initialize location and velocity
        # X = np.random.randn(n, self.dim) * self.loc_std  # N(0, loc_std)
        X = np.random.randn(n, self.dim) * self.loc_std  # N(0, loc_std)
        V = np.random.randn(n, self.dim)  # N(0, 1)
        v_norm = np.sqrt((V ** 2).sum(axis=-1)).reshape(-1, 1)
        V = V / v_norm * self.vel_norm

        # initialize physical objects
        self.physical_objects = []
        # node_idx = 0
        selected = []
        # for _ in range(n_isolated):
        #     rest = [idx for idx in range(self.n_balls) if idx not in selected]
        #     node_idx = list(np.random.choice(rest, size=1, replace=False))
        #     current_obj = Isolated(n_balls=1, node_idx=node_idx,
        #                            charge=[charges[node_idx[0]]], type='Isolated')
        #     selected.extend(node_idx)
        #
        #     # current_obj = Isolated(n_balls=1, node_idx=[node_idx],
        #     #                        charge=[charges[node_idx]], type='Isolated')
        #     self.physical_objects.append(current_obj)
        #     # node_idx += 1
        #
        # for _ in range(n_stick):
        #     rest = [idx for idx in range(self.n_balls) if idx not in selected]
        #     node_idx = list(np.random.choice(rest, size=2, replace=False))
        #     current_obj = Stick(n_balls=2, node_idx=node_idx,
        #                         charge=[charges[node_idx[0]], charges[node_idx[1]]], type='Stick')
        #     selected.extend(node_idx)
        #
        #     # current_obj = Stick(n_balls=2, node_idx=[node_idx, node_idx + 1],
        #     #                     charge=[charges[node_idx], charges[node_idx + 1]], type='Stick')
        #     self.physical_objects.append(current_obj)
        #     # node_idx += 2
        #
        # for _ in range(n_hinge):
        #     rest = [idx for idx in range(self.n_balls) if idx not in selected]
        #     node_idx = list(np.random.choice(rest, size=3, replace=False))
        #     current_obj = Hinge(n_balls=3, node_idx=node_idx,
        #                         charge=[charges[node_idx[0]], charges[node_idx[1]], charges[node_idx[2]]], type='Hinge')
        #     selected.extend(node_idx)
        #
        #     # current_obj = Hinge(n_balls=3, node_idx=[node_idx, node_idx + 1, node_idx + 2],
        #     #                     charge=[charges[node_idx], charges[node_idx + 1], charges[node_idx + 2]], type='Hinge')
        #     self.physical_objects.append(current_obj)
        #     # node_idx += 3

        for _ in range(n_complex):
            size = rand_size[_]
            rest = [idx for idx in range(self.n_balls) if idx not in selected]
            node_idx = list(np.random.choice(rest, size=size, replace=False))
            current_obj = Complex(n_balls=size, node_idx=node_idx, charge=[charges[node_idx[i]] for i in range(size)],
                                  type='Complex')
            selected.extend(node_idx)
            self.physical_objects.append(current_obj)

        assert n == self.n_balls == len(selected) == len(np.unique(selected))

        assert len(self.physical_objects) == X_c.shape[0]
        # check and adjust initial conditions
        for idx, obj in enumerate(self.physical_objects):
            # X, V = obj.initialize(X, V)
            X, V = obj.initialize(X, V, X_c=X_c[idx], rr=min_d)

        # book-keeping x and v
        self.X, self.V = X, V

    @staticmethod
    def _l2(A, B):
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def compute_F(self, X, V):
        n = self.n_balls
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(X, X), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * self.edges / l2_dist_power3
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[self.diag_mask]).min() > 1e-10)

            # here for minor precision issue with respect to the original script
            _X = X.T
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(_X[0, :],
                                       _X[0, :]).reshape(1, n, n),
                     np.subtract.outer(_X[1, :],
                                       _X[1, :]).reshape(1, n, n),
                     np.subtract.outer(_X[2, :],
                                       _X[2, :]).reshape(1, n, n)))).sum(axis=-1)
            F = F.T

            # adjust F scale
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

        return F

    def simulate_one_step(self):
        X, V = self.X, self.V
        F = self.compute_F(X, V)
        for obj in self.physical_objects:
            X, V = obj.update(X, V, F, self.delta_t)
        self.X, self.V = X, V
        return X, V

    def check(self):
        for obj in self.physical_objects:
            obj.check(self.X, self.V)

    def is_valid(self):
        if self.box_size:
            return np.all(self.X <= self.box_size) and np.all(self.X >= - self.box_size)
        else:
            if (self.V > 5).any():
                return False
            return True  # no box size

    def configuration(self):
        cfg = {}
        for obj in self.physical_objects:
            _type = obj.type
            _node_idx = obj.node_idx
            if _type in cfg:
                cfg[_type].append(_node_idx)
            else:
                cfg[_type] = [_node_idx]
        return cfg

    def print(self):
        print('X:')
        print(self.X)
        print('V:')
        print(self.V)


def visualize():
    np.random.seed(89)
    sizes = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    system = System(n_isolated=0, n_complex=len(sizes), complex_sizes=sizes, n_hinge=0, n_stick=0)
    steps = 5001
    ret = []
    for step in tqdm(range(steps)):
        system.simulate_one_step()
        ret.append((system.X.copy(), system.V.copy()))
        system.check()
    print(len(ret))
    cfg = system.configuration()
    print(cfg)
    # exit(0)
    edges = []
    # construct edges
    for obj_type in cfg:
        if obj_type == 'Isolated':
            pass
        elif obj_type == 'Complex':
            nodes = cfg[obj_type]
            # put together
            _nodes = []
            for n in nodes:
                _nodes += n
            nodes = _nodes

            # print(nodes)
            split = sizes
            st = 0
            for sp in split:
                cur_nodes = nodes[st: st + sp]
                # fully connected
                for i in range(len(cur_nodes)):
                    for j in range(len(cur_nodes)):
                        if i != j:
                            edges.append([cur_nodes[i], cur_nodes[j]])
                st = st + sp
        else:
            raise NotImplementedError('Unknown object type:', obj_type)
    print(edges)
    # exit(0)
    x_start = ret[1100][0]
    print(x_start)
    print(x_start.shape)
    # exit(0)
    x_center = np.mean(x_start[..., 0])
    y_center = np.mean(x_start[..., 1])
    z_center = np.mean(x_start[..., 2])

    fig = plt.figure()
    ax = Axes3D(fig)
    figure_3D_size = 8

    ax.set_xlim3d(z_center - figure_3D_size / 2, z_center + figure_3D_size / 2)
    ax.set_ylim3d(x_center - figure_3D_size / 2, x_center + figure_3D_size / 2)
    ax.set_zlim3d(y_center - figure_3D_size / 2, y_center + figure_3D_size / 2)

    N_NODE = sum(sizes)
    # plot the start position
    xs, ys, zs = [], [], []
    for i in range(N_NODE):
        xs.append(x_start[i][0])
        ys.append(x_start[i][1])
        zs.append(x_start[i][2])
    plt.plot(zs, xs, ys, 'b.')
    for edge in edges:
        xs = [x_start[edge[0]][0], x_start[edge[1]][0]]
        ys = [x_start[edge[0]][1], x_start[edge[1]][1]]
        zs = [x_start[edge[0]][2], x_start[edge[1]][2]]
        plt.plot(zs, xs, ys, 'b')
    # plot the end position
    xs, ys, zs = [], [], []
    x_end = ret[2600][0]
    print(x_end)
    for i in range(N_NODE):
        xs.append(x_end[i][0])
        ys.append(x_end[i][1])
        zs.append(x_end[i][2])
    plt.plot(zs, xs, ys, 'r.')
    for edge in edges:
        xs = [x_end[edge[0]][0], x_end[edge[1]][0]]
        ys = [x_end[edge[0]][1], x_end[edge[1]][1]]
        zs = [x_end[edge[0]][2], x_end[edge[1]][2]]
        plt.plot(zs, xs, ys, 'red')
    plt.show()

    print(np.sum((x_start[0] - x_start[1]) ** 2))
    print(np.sum((x_end[0] - x_end[1]) ** 2))


def test():
    np.random.seed(10)
    # system = System(n_isolated=4, n_stick=0, n_hinge=2)
    # n_balls = 20

    # system = System(n_isolated=10, n_stick=5, n_hinge=0)
    # np.random.seed(10)
    # system.X = np.random.rand(20, 3)
    # system.V = np.random.rand(20, 3)
    # charges = np.random.choice([1, -1], size=20).reshape(-1, 1)
    # system.edges = charges.dot(charges.transpose())
    # system.charges = charges
    # for obj in system.physical_objects:
    #     system.X, system.V = obj.initialize(system.X, system.V)

    # system.X = np.array([
    #     [-0.4400, 1.8563, -0.8407],
    #     [ 1.8749,  0.8352,  0.7475],
    #     [ 1.8236, -0.2337,  0.8648],
    #     [0.2393, 0.3604, 0.9857],
    #     [-0.9331, -1.2261,  2.5555],
    #     ])
    # system.V = np.array([
    #     [-0.1481, 0.1102, -0.1308],
    #     [ 0.6803,  0.1725, -0.2684],
    #     [ 0.2594,  0.2284,  0.0572],
    #     [1.0518, -0.3935, 0.7333],
    #     [-0.3815,  0.0661,  0.1284],
    #     ])
    # charges = np.array([-1, 1, 1, 1, 1]).reshape(-1, 1)
    # system.edges = charges.dot(charges.transpose())
    # system.charges = charges
    #
    # system.print()
    #
    # for obj in system.physical_objects:
    #     system.X, system.V = obj.initialize(system.X, system.V)
    # system.check()

    system = System(n_isolated=3, n_stick=5, n_hinge=0, n_complex=2, complex_sizes=[3, 4])

    system.print()
    steps = 5001
    ret = []
    for step in tqdm(range(steps)):
        system.simulate_one_step()
        ret.append((system.X.copy(), system.V.copy()))
        # system.check()
    system.print()
    # print(system.X.dtype)
    # print(system.charges)
    # print(system.is_valid())
    return ret


if __name__ == '__main__':
    # for i, F in enumerate()
    # ret = test()
    visualize()


