import numpy as np

from octree.octree_node import OctreeNode
from octree.sampled_boxes import SampledAABB, SampledBBox


class OctreeRoot():
    RNG = np.random.default_rng(0)

    def __init__(self, tree_root: OctreeNode) -> None:
        self._tree_root = tree_root
        self.update()

    """ ---------------------------- """
    """  >>>>> Public methods <<<<<  """
    """ ---------------------------- """
    """ Containment checks """
    def contains_point(self, point: np.ndarray) -> bool:
        return self._tree_root.contains_point(point)

    def contains_box(self, bbox: SampledBBox) -> bool:
        """Must contain entire bbox

        Args:
            bbox (SampledBBox): The bbox to check

        Returns:
            bool: True if entire bbox is contained, False otherwise.
        """
        for point in bbox.sample_edges_by_distance(self._tree_root.inner_radius / 2):
            if not self.contains_point(point):
                return False
        return True

    def contains_sphere(self, center: np.ndarray, radius: float) -> bool:
        return self._tree_root.contains_sphere(center, radius)

    def find_containing_leaf(self, point: np.ndarray) -> Optional[OctreeNode]:
        return self._tree_root.find_containing_leaf(point)

    """ Closest checks """
    def find_closest_leaf(self, point: np.ndarray) -> Optional[OctreeNode]:
        return self._tree_root.find_closest_leaf(point)

    def find_closest_leaf_to_aabb(self, aabb: SampledAABB) -> Optional[OctreeNode]:
        return self._tree_root.find_closest_leaf_to_bbox(aabb)

    """ Interaction with other trees """
    def merge(self, other: Union['OctreeRoot', OctreeNode]) -> None:
        if isinstance(other, OctreeRoot):
            other_root = other.root
        else:
            other_root = other
        # TODO: Merge another octree, adding all points into this tree.
        raise NotImplementedError()

    """ Self manipulation """
    def trim_empty(self) -> None:
        # TODO: Trim empty nodes from the whole tree
        raise NotImplementedError()

    def expand_all(self, up_to_depth: int = 0) -> None:
        # TODO: Expand all nodes in the tree
        raise NotImplementedError()

    def update(self) -> None:
        self._zorder_dict = self._tree_root.gather_zorder_dict()
        self._leaves = self._tree_root.gather_leaves()

    """ Random sampling """
    def sample_points(self, n_samples: int = 1, rng=None) -> np.ndarray:
        if rng is None:
            rng = self.RNG

        n_leaves = len(self._leaves)
        if n_leaves == 0:
            return []

        leaf_indices = rng.choice(n_leaves, n_samples)
        samples = []
        for leaf_idx in leaf_indices:
            leaf = self._leaves[leaf_idx]
            min_bounds, max_bounds = leaf.get_bounds()
            # min_bounds, max_bounds = self.root.get_bounds()
            samples.append(self.random_point_in_aabb(min_bounds, max_bounds, rng=rng))

        return np.array(samples)

    """ ---------------------------- """
    """  >>>>> Debug methods <<<<<<  """
    """ ---------------------------- """
    def print_stats(self) -> None:
        print(self._tree_root.print_stats())

    """ ---------------------------- """
    """  >>>>> Magic methods <<<<<<  """
    """ ---------------------------- """
    def __getitem__(self, zorder: int) -> OctreeNode:
        return self._zorder_dict[zorder]

    def __contains__(self, zorder: int) -> bool:
        return zorder in self._zorder_dict

    """ ---------------------------- """
    """   >>>>>> Properties <<<<<<   """
    """ ---------------------------- """
    @property
    def root(self) -> OctreeNode:
        return self._tree_root

    @property
    def leaves(self) -> list[OctreeNode]:
        return self._leaves

    """ ---------------------------- """
    """ >>>>>> Class methods <<<<<<  """
    """ ---------------------------- """
    @classmethod
    def random_point_in_aabb(cls, min_bounds, max_bounds, rng=None) -> np.ndarray:
        if rng is None:
            rng = cls.RNG
        return rng.uniform(min_bounds, max_bounds)

    @classmethod
    def set_rng_seed(cls, seed: int) -> None:
        cls.RNG = np.random.default_rng(seed)

    """ ---------------------------- """
    """ >>>>>> Static methods <<<<<< """
    """ ---------------------------- """
    @classmethod
    def create_from_occupancy_tensor(cls, mat: np.ndarray, leaf_scale: float, center: np.ndarray, allow_empty_above_depth: int = 0) -> 'OctreeRoot':
        return cls(OctreeNode.generate_from_occupancy_tensor(mat, leaf_scale, center, allow_empty_above_depth))

    @classmethod
    def create_dense(cls, root_depth: int, leaf_scale: float, center: np.ndarray, up_to_depth: int = 0) -> 'OctreeRoot':
        return cls(OctreeNode.generate_dense(root_depth, leaf_scale, center, up_to_depth))

    @staticmethod
    def convert_min_bounds_to_center(min_bounds: np.ndarray, matrix, leaf_scale) -> np.ndarray:
        def check_max_depth(shape, d=0):
            r = shape / 2
            if np.any(r > 1):
                return check_max_depth(r, d + 1)
            else:
                return d + 1

        max_d = check_max_depth(np.r_[matrix.shape])
        return min_bounds + (2**max_d * leaf_scale) / 2