from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from mayavi import mlab
from sympy import false


class OctreeNode:

    def __init__(self,
                 zorder: int,
                 depth: int,
                 scale: float,
                 grid_pos: np.ndarray,
                 center: np.ndarray,
                 parent: Optional['OctreeNode'] = None,
                 children: Optional[list['OctreeNode']] = None,
                 value: bool = False,
                 ) -> None:
        """Creates an oct tree node.

        Args:
            zorder (int): Z-order index of the node. Root should have negative z-order. The remaining nodes should have local z-order according to their grid position ([0, 0, 0] == 0, [1, 0, 0] == 1, [0, 1, 0] == 2, ...). Starting z-order is taken from parent. E.g., if parent has z-order 23 and a child has local z-order 3, the z-order of the child is 23 * 10 + 3 = 233.
            depth (int): Node depth. Depth is reversed from typical tree. Leaves have depth 0. Parent nodes have increasing depth, with root having the highest depth.
            scale (float): Metric scale of the (hypothetical) leaf node. Leaf scale represents side length of the cube represented by a leaf node. This is used to compute local scale. E.g., if the leaf scale is 0.05 (5cm) and current depth is 2 (third layer from the leaf), the scale of the node is 0.05 * 2**2 = 0.25.
            grid_pos (np.ndarray): position of the node in parent's grid space. This is a pseudo-binary ordering (first node->min=[0,0,0], eighth node->max=[1,1,1]). [0,0,0] means first row, first column, first depth (position along the third dimension). [0,0,1] is second column, first row, first depth, etc.
            center (np.ndarray): Position of the node's center in world space (metric).
            parent (Optional[&#39;OctreeNode&#39;], optional): Link to the parent node. Defaults to None. (if root, parent is None)
            children (Optional[list[OctreeNode]], optional): List of child nodes. Defaults to None. If the node is leaf, children is None. However, any terminal node can have empty children list. Terminal means there are no more children but does not have to be leaf. E.g., if all children nodes would be empty and empty nodes are not allowed below current depth than this node will exist but will have no children = terminal.
            value (bool, optional): Value of the node. Defaults to True. True means occupied. False means free. For non-leaf node, True means some descendant is occupied. False means there is no occupied descendant.
        """
        self._value: bool = value
        self._zorder: int = zorder
        self._grid_pos: np.ndarray = grid_pos
        self._center: np.ndarray = center
        self._parent: Optional['OctreeNode'] = parent

        params = OctreeNode.compute_for_depth(depth, scale, center)
        self._depth: int = depth
        self._n_cells_per_dim: int = params["n_cells_per_dim"]
        self._scale: float = params["scale"]
        self._inner_radius: float = params["inner_radius"]
        self._inner_radius_sq: float = params["inner_radius"]**2
        self._outer_radius: float = params["outer_radius"]
        self._outer_radius_sq: float = params["outer_radius"]**2
        # self._corners: np.ndarray = params["corners"] + self._center

        self._children: Optional[list['OctreeNode']]
        self._virtual_children: Optional[list['OctreeNode']] = None

        self.set_children(children)

    @staticmethod
    def compute_corners(radius: float, center: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
        corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) * radius * 2 - radius
        return corners + center

    @staticmethod
    def compute_for_depth(depth: int, scale: float, center: np.ndarray = np.array([0, 0, 0])) -> dict[str, Any]:
        params = {
            "depth": depth,
            "n_cells_per_dim": 2 ** depth,
        }
        params["scale"] = scale * params["n_cells_per_dim"]
        params["inner_radius"] = params["scale"] / 2
        params["outer_radius"] = np.sqrt((params["scale"] / 2) ** 2 * 2)
        # params["corners"] = OctreeNode.compute_corners(params["inner_radius"], center)
        return params

    def _compute_positions(self) -> None:
        # TODO: compute correct positions
        if self._children is None:
            return
        for child in self._children:
            child._compute_positions()

    def set_children(self, children: Optional[list['OctreeNode']]) -> None:
        if children is not None and len(children) > 0:
            children = sorted(children, key=lambda node: node.zorder)
            self._children_dict = {node.zorder: node for node in children}
        self._children = children
        self._n_children: int = len(self._children) if self._children is not None else 0

    def is_leaf(self) -> bool:
        return self._depth == 0

    def is_terminal(self) -> bool:
        return self.is_leaf() or self._children is None or self._n_children == 0

    def count_descendants(self) -> int:
        if self.is_terminal():
            return 1
        return sum([child.count_descendants() for child in self._children])  # type: ignore

    def count_occupied(self) -> int:
        if self._children is None:
            return 1 if self.value else 0
        return sum([child.count_occupied() for child in self._children])

    def make_empty(self) -> None:
        self.value = False
        if self._children is not None:
            for child in self._children:
                child.make_empty()

    def make_occupied(self) -> None:
        self.value = True
        if self.is_leaf():
            for child in self._children:  # type: ignore
                child.make_occupied()

    def check_occupancy(self) -> bool:
        if not self.is_leaf():
            self.value = any([child.check_occupancy() for child in self._children])  # type: ignore
        return self.value

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        min_bounds = self._center - self._inner_radius
        max_bounds = self._center + self._inner_radius
        return min_bounds, max_bounds

    def gather_leaves(self) -> list['OctreeNode']:
        if self.is_leaf():
            return [self]
        elif self._children is None:
            return []
        else:
            result = []
            for child in self._children:
                result.extend(child.gather_leaves())
            return result

    def gather_zorder_dict(self) -> dict[int, 'OctreeNode']:
        if self.is_terminal():
            return {}
        else:
            result = self._children_dict
            for child in self._children:
                result.update(child.gather_zorder_dict())
            return result

    def __getitem__(self, key):
        if key in self._children_dict:
            return self._children_dict[key]
        else:
            new_key = self.zorder * 10 + key
            if new_key in self._children_dict:
                return self._children_dict[new_key]
            else:
                return None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def parent(self):
        return self._parent

    @property
    def depth(self):
        return self._depth

    @property
    def zorder(self):
        return self._zorder

    @property
    def n_cells_per_dim(self):
        return self._n_cells_per_dim

    @property
    def scale(self):
        return self._scale

    @property
    def grid_pos(self):
        return self._grid_pos

    @property
    def center(self):
        return self._center

    @property
    def inner_radius(self):
        return self._inner_radius

    @property
    def outer_radius(self):
        return self._outer_radius

    @property
    def children(self):
        return self._children

    @property
    def virtual_children(self):
        return self._children

    @property
    def all_children(self):
        return (self._children if self._children is not None else []) + (self._virtual_children if self._virtual_children is not None else [])

    @property
    def n_children(self):
        return self._n_children

    @property
    def corners(self):
        return self._corners

    def _print_children__(self, tabs: int) -> str:
        if self._children is None:
            return "[leaf]"
        return "\n" + "\n ".join([child.pretty_print(tabs) for child in self._children])

    def pretty_print(self, tabs: int = 0) -> str:
        tab = "\t" * tabs
        return f"{tab}OctreeNode(z-order={self._zorder}, depth={self._depth}, origin={self._center}, pos={self._grid_pos}, inner_radius={self._inner_radius}, outter_radius={self._outer_radius}, children[{self._n_children}]:{self._print_children__(tabs + 1) if self._children is not None else '[leaf]'})"

    def print_stats(self) -> str:
        return f"Depth: {self._depth}\nTotal cells: {self.count_descendants()}\nOccupied cells: {self.count_occupied()}"

    def __repr__(self) -> str:
        return f"OctreeNode(z-order={self._zorder}, depth={self._depth}, origin={self._center}, pos={self._grid_pos}, inner_radius={self._inner_radius}, outter_radius={self._outer_radius}, n_children={self._n_children})"

    def contains_point(self, point: np.ndarray) -> bool:
        if self.point_possibly_near(point) and self.point_within(point):
            if self.is_leaf():
                return True
            if self._children is None: # only count points if they are within nodes
                # TODO: expand non-leaf nodes
                return False
            return all(child.contains_point(point) for child in self._children)
        else:
            return False

    def find_closest_leaf(self, point: np.ndarray) -> Optional['OctreeNode']:
        if self.contains_point(point):
            if self.is_leaf():
                return self
            if self._children is not None:
                result_children = [child.find_closest_leaf(point) for child in self._children]
                if any(result_children):
                    return next((c for c in result_children if c is not None))

    def expand_non_leaf(self) -> None:
        if self.is_leaf():
            return
        # TODO:

    def point_within(self, point: np.ndarray) -> bool:
        return self._point_in_box(point, self._center, self._inner_radius)

    def point_possibly_near(self, point: np.ndarray) -> bool:
        return self._point_in_sphere_sq(point, self._center, self._outer_radius_sq)

    @staticmethod
    def _point_in_box(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
        off_center_diff = np.abs(point - center)
        return all(off_center_diff < radius)

    @staticmethod
    def _point_in_sphere_sq(point: np.ndarray, center: np.ndarray, radius_squared: float) -> bool:
        distance = (point[0] - center[0])**2 + (point[1] - center[1])**2 + (point[2] - center[2])**2
        return distance < radius_squared

    @staticmethod
    def _sphere_in_sphere_sq(center_a: np.ndarray, center_b: np.ndarray, radius_a_squared: float, radius_b_squared: float = 0) -> bool:
        distance = np.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2 + (center_a[2] - center_b[2])**2)
        return distance < radius_a_squared + radius_b_squared

    @staticmethod
    def _generate_octree_recursive(tensor: Optional[np.ndarray],
                                   parent: Optional['OctreeNode'],
                                   zorder: int,
                                   depth: int,
                                   grid_pos: np.ndarray,
                                   origin: np.ndarray,
                                   leaf_scale: float,
                                   allow_empty_above_depth: int,
                                   ) -> 'OctreeNode':
        node_occupied = tensor is not None and tensor.sum() > 0
        node = OctreeNode(-1 if parent is None else zorder * 10, depth, leaf_scale, grid_pos, origin, parent, value=node_occupied)
        if depth == 0:
            return node

        division_factor = 2 ** (depth - 1)
        inner_radius_child = (leaf_scale * (2 ** depth)) / 2

        children = []
        shape = None if tensor is None else np.r_[tensor.shape]
        sub_tensor = None
        desc_zorder = zorder * 10
        sub_space_occupied_any = False
        sub_space_occupied = False

        for r in range(2):
            for c in range(2):
                for d in range(2):
                    if tensor is not None:  # tensor is (typically) None if generating dense tree (without occupancy tensor)
                        sub_tensor = tensor[r * division_factor:min(shape[0], (r + 1) * division_factor), c * division_factor:min(shape[1], (c + 1) * division_factor), d * division_factor:min(shape[2], (d + 1) * division_factor)]  # type: ignore
                        sub_space_occupied = sub_tensor.sum() > 0
                        sub_space_occupied_any = sub_space_occupied_any or sub_space_occupied
                    if allow_empty_above_depth < depth or sub_space_occupied:
                        child_zorder = desc_zorder + r * 4 + c * 2 + d
                        grid_pos_child = np.r_[r, c, d]
                        origin_child = origin + grid_pos_child * inner_radius_child * 2 - inner_radius_child
                        children.append(OctreeNode._generate_octree_recursive(sub_tensor, node, child_zorder, depth - 1, grid_pos_child, origin_child, leaf_scale, allow_empty_above_depth))

        node.set_children(children)
        node.value = sub_space_occupied_any

        return node

    @classmethod
    def generate_from_occupancy_tensor(cls, dense_tensor: np.ndarray, leaf_scale: float, center: np.ndarray, allow_empty_above_depth: int = 0) -> 'OctreeNode':
        """
        Generate an octree from the given dense tensor with the specified depth threshold.

        Parameters:
        - dense_tensor (np.ndarray): the dense tensor from which to generate the octree
        - depth_threshold (int): the depth threshold for octree generation

        Returns:
        - octree (OctreeNode): the generated octree
        """
        shape = np.r_[dense_tensor.shape]

        def check_max_depth(shape, d=0):
            r = shape / 2
            if np.any(r > 1):
                return check_max_depth(r, d + 1)
            else:
                return d + 1

        depth = check_max_depth(shape)

        return cls._generate_octree_recursive(dense_tensor, None, 0, depth, np.r_[0, 0, 0], center, leaf_scale, allow_empty_above_depth)

    @classmethod
    def generate_dense(cls, root_depth: int, leaf_scale: float, center: np.ndarray, up_to_depth: int = 0) -> 'OctreeNode':
        return cls._generate_octree_recursive(None, None, 0, root_depth, np.r_[0, 0, 0], center, leaf_scale, up_to_depth)
# mamba install nvidia/label/cuda-11.4.2::cuda-toolkit nvidia/label/cuda-11.4.2::cuda-nvcc nvidia/label/cuda-11.4.2::libcublas nvidia/label/cuda-11.4.2::libcublas-dev nvidia/label/cuda-11.4.2::nsight-compute nvidia/label/cuda-11.4.2::cuda-cudart nvidia/label/cuda-11.4.2::cuda-cudart-dev nvidia/label/cuda-11.4.2::cuda-libraries-dev


nx_node_type = tuple[int, int]


class OTDrawer():

    def __init__(self, tree_root: OctreeNode, exclude_empty: bool = False) -> None:
        self._tree_root = tree_root
        self._graph = self._convert_to_networkx(exclude_empty)

    def _convert_to_networkx(self, exclude_empty: bool) -> nx.DiGraph:
        def _convert_oct2nx(oct_node: OctreeNode) -> tuple[nx_node_type, dict]:
            return (oct_node.depth, oct_node.zorder), {"oc_node": oct_node}

        # def _convert_to_networkx_recursive(parent_nx_node: tuple[tuple[int, np.ndarray], dict], oc_node: OctreeNode) -> tuple[tuple[int, np.ndarray], dict]:
        def _convert_to_networkx_recursive(nx_node: tuple[nx_node_type, dict]) -> nx.DiGraph:
            oct_node = nx_node[1]["oc_node"]
            g = nx.DiGraph()
            g.add_nodes_from([nx_node])
            if oct_node.is_leaf():
                return g

            for oct_child in oct_node.children:
                if exclude_empty and not oct_child.value:
                    continue
                nx_child = _convert_oct2nx(oct_child)
                g.add_node(nx_child[0], **nx_child[1])
                g.add_edge(nx_node[0], nx_child[0])
                g = nx.compose(g, _convert_to_networkx_recursive(nx_child))
            return g

        nx_root = _convert_oct2nx(self.tree_root)
        return _convert_to_networkx_recursive(nx_root)

    def render(self) -> None:
        indexed_graph = nx.convert_node_labels_to_integers(self.graph)
        scalars, xyz = zip(*[(n[0][0] + 1, n[1]["oc_node"].center) for n in self.graph.nodes(data=True)])
        xyz = np.array(xyz)

        mlab.figure()

        pts = mlab.points3d(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            scalars,
            scale_factor=0.02,
            # scale_mode="none",
            scale_mode="scalar",
            colormap="Blues",
            mode="cube",
            resolution=8,
            opacity=0.9
        )

        pts.mlab_source.dataset.lines = np.array(list(indexed_graph.edges()))
        tube = mlab.pipeline.tube(pts, tube_radius=0.005)
        mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
        mlab.orientation_axes()
        mlab.show()
        # nx.draw(self.graph, with_labels=True)
        # plt.show()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def tree_root(self) -> OctreeNode:
        return self._tree_root

    def pretty_print(self) -> str:
        return self.tree_root.pretty_print()


class OctreeRoot():
    RNG = np.random.default_rng(0)

    def __init__(self, tree_root: OctreeNode) -> None:
        self._tree_root = tree_root
        self.update()

    def add_point(self, point: np.ndarray) -> None:
        # TODO: Add point to octree. Find closest leaf and make the leaf occupied. If the leaf does not exist yet, add it
        pass

    def remove_point(self, point: np.ndarray) -> None:
        # TODO: Remove point from octree
        pass

    def contains_point(self, point: np.ndarray) -> bool:
        return self._tree_root.contains_point(point)

    def contains_box(self, bb_points: np.ndarray) -> bool:
        for point in bb_points:
            if not self.contains_point(point):
                return False
        return True

    def contains_sphere(self, center: np.ndarray, radius: float) -> bool:
        return self._tree_root.contains_sphere(center, radius)

    def sample_point(self, *args, **kwargs) -> np.ndarray:
        return

    def find_closest_leaf(self, point: np.ndarray) -> Optional[OctreeNode]:
        return self._tree_root.find_closest_leaf(point)

    def merge(self, other: Union['OctreeRoot', OctreeNode]) -> None:
        if isinstance(other, OctreeRoot):
            other_root = other.tree_root
        else:
            other_root = other
        # TODO: Merge another octree, adding all points into this tree.

    def trim_empty(self) -> None:
        # TODO: Trim empty nodes from the whole tree
        pass

    def expand_all(self, up_to_depth: int = 0) -> None:
        # TODO: Expand all nodes in the tree
        pass

    def update(self) -> None:
        self._zorder_dict = self._tree_root.gather_zorder_dict()
        self._leaves = self._tree_root.gather_leaves()

    @classmethod
    def random_point_in_aabb(cls, min_bounds, max_bounds, rng=None) -> np.ndarray:
        if rng is None:
            rng = cls.RNG
        return rng.uniform(min_bounds, max_bounds)

    def sample(self, n_samples: int = 1, rng=None) -> np.ndarray:
        if rng is None:
            rng = self.RNG

        leaf_idx = rng.choice(len(self._leaves), n_samples)
        sampled_leaves = self._leaves[leaf_idx]
        if n_samples == 1:
            min_bounds, max_bounds = sampled_leaves[0].get_bounds()
            return self.random_point_in_aabb(min_bounds, max_bounds, rng=rng)
        else:
            # TODO: rework to work with leaves
            corners_multi = np.apply_along_axis(self._get_cell_corners, 1, sample_cells, self.origin, self._scale)
            return np.apply_along_axis(self.random_point_in_bounding_box, 1, corners_multi, rng=rng)

    def __getitem__(self, zorder: int) -> OctreeNode:
        return self._zorder_dict[zorder]

    def __contains__(self, zorder: int) -> bool:
        return zorder in self._zorder_dict

    def print_stats(self) -> None:
        self._tree_root.print_stats()

    @property
    def root(self) -> OctreeNode:
        return self._tree_root

    @classmethod
    def create_from_occupancy_tensor(cls, mat: np.ndarray, leaf_scale: float, center: np.ndarray, allow_empty_above_depth: int = 0) -> 'OctreeRoot':
        return cls(OctreeNode.generate_from_occupancy_tensor(mat, leaf_scale, center, allow_empty_above_depth))

    @classmethod
    def create_dense(cls, root_depth: int, leaf_scale: float, center: np.ndarray, up_to_depth: int = 0) -> 'OctreeRoot':
        return cls(OctreeNode.generate_dense(root_depth, leaf_scale, center, up_to_depth))


if __name__ == "__main__":
    # mat = np.zeros((7, 9, 7), dtype=np.uint8)
    # mat[2:6, 2:6, 2:6] = 1
    # mat[6, 8, 6] = 1

    side = 64
    root_depth = 7
    allow_empty_above_depth = 5
    draw_empty = True
    mat = (OctreeRoot.RNG.random(size=(side, side, side)) > 0.999).astype(int)

    octree = OctreeRoot.create_from_occupancy_tensor(mat, 0.05, np.array([1, 1, 1]), allow_empty_above_depth=allow_empty_above_depth)
    # octree = OctreeRoot.create_dense(root_depth, 0.05, np.array([1, 1, 1]), up_to_depth=allow_empty_above_depth)

    print(octree.print_stats())
    # print(octree.pretty_print())
    drawer = OTDrawer(octree.root, exclude_empty=not draw_empty)
    # print(list(drawer.graph.nodes(data=True)))

    drawer.render()
