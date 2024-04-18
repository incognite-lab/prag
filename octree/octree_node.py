from typing import Any, ClassVar, Optional, Union

import numpy as np

from octree.sampled_boxes import SampledBBox


class OctreeNode:
    OCCUPANCY_MODE_BOOLEAN: ClassVar[int] = 1
    OCCUPANCY_MODE_ITEMS: ClassVar[int] = 2
    OCCUPANCY_MODE: ClassVar[int] = OCCUPANCY_MODE_BOOLEAN

    def __init__(self,
                 zorder: int,
                 depth: int,
                 scale: float,
                 grid_pos: np.ndarray,
                 center: np.ndarray,
                 parent: Optional['OctreeNode'] = None,
                 children: Optional[list['OctreeNode']] = None,
                 value: Union[bool, list[Any]] = False,
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
            value (bool, optional): Value of the node. Defaults to True. True means occupied. False means free. For non-leaf node, True means some descendant is occupied. False means there is no occupied descendant. Could also be a value or list of values, if occupancy mode is set to items. If it is False or None and occupancy mode is set to items, the value will be an empty list.
        """
        self._occupancy_mode = OctreeNode.OCCUPANCY_MODE
        assert self._occupancy_mode in [OctreeNode.OCCUPANCY_MODE_BOOLEAN, OctreeNode.OCCUPANCY_MODE_ITEMS], f"Unknown occupancy mode {self._occupancy_mode}. See OctreeNode.OCCUPANCY_MODE_*"
        assert parent is None or self._occupancy_mode == parent._occupancy_mode, f"Node occupancy mode is {self._occupancy_mode} but parent is {parent._occupancy_mode}. Single tree cannot have different occupancy mode. Mode may have changed during tree generation (it should not)."
        if self._occupancy_mode == OctreeNode.OCCUPANCY_MODE_ITEMS:
            if not isinstance(value, list):
                if value is False or value is None:  # False or None means free
                    value = []
                else:
                    value = [value]
        self._value: Union[bool, list[Any]] = value

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

        def _find_root():
            root = self
            while root._parent is not None:
                root = root._parent
            return root

        self.__root = _find_root()

    @staticmethod
    def compute_corners(radius: float, center: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
        corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) * radius * 2 - radius
        return corners + center

    @staticmethod
    def compute_for_depth(depth: int, scale: float, center: np.ndarray = np.array([0, 0, 0])) -> dict[str, Any]:
        """Computes parameters for the given depth and scale.

        Args:
            depth (int): The depth for which to compute the parameters.
            scale (float): The leaf scale (size of the leaf cell, metric).
            center (np.ndarray, optional): The center of the octree. Defaults to np.array([0, 0, 0]).

        Returns:
            dict[str, Any]: Dictionary containing the computed parameters.
        """
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
        """Computes the positions of the children nodes.
        """
        # TODO: compute correct positions
        if self._children is None:
            return
        for child in self._children:
            child._compute_positions()

    def set_children(self, children: Optional[list['OctreeNode']]) -> None:
        """Sets the children of this node.

        Args:
            children (Optional[list[OctreeNode]]): List of child nodes.
        """
        if children is not None and len(children) > 0:
            children = sorted(children, key=lambda node: node.zorder)
            self._children_dict = {node.zorder: node for node in children}
        self._children = children
        self._n_children: int = len(self._children) if self._children is not None else 0

    def is_leaf(self) -> bool:
        """Checks if this node is a leaf.

        Returns:
            bool: True if this node is a leaf, False otherwise.
        """
        return self._depth == 0

    def is_terminal(self) -> bool:
        """Checks if this node is a terminal node. That is, it has no children. May or may not be a leaf.

        Returns:
            bool: True if this node is a terminal node, False otherwise.
        """
        return self.is_leaf() or self._children is None or self._n_children == 0

    def count_descendants(self) -> int:
        """Counts the number of descendants (all nodes below this node).

        Returns:
            int: The number of descendants.
        """
        if self.is_terminal():
            return 1
        return sum([child.count_descendants() for child in self._children])  # type: ignore

    def count_occupied(self) -> int:
        """Counts the number of occupied cells in the subtree below this node.

        Returns:
            int: The number of occupied cells.
        """
        if self._children is None:
            return 1 if self.value else 0
        return sum([child.count_occupied() for child in self._children])

    def make_empty(self) -> None:
        """Makes this node empty, that is, not occupied. All children are made empty as well.
        """
        self.__empty_node()
        if self._children is not None:
            for child in self._children:
                child.make_empty()

    def __empty_node(self) -> None:
        """Makes this node empty, that is, not occupied. Do not call directly as this does not propagate to children.
        Call 'make_empty' instead.
        """
        if self._occupancy_mode == OctreeNode.OCCUPANCY_MODE_ITEMS:
            self.value = []
        else:
            self.value = False

    def make_occupied(self, value: Any = None) -> None:
        """Makes this node occupied. All children are made occupied as well.

        Args:
            value (Any, optional): Value to be stored in the node if occupancy mode is 'items'. Defaults to None.
        """
        self.__occupy_node(value)
        if self.is_leaf():
            for child in self._children:  # type: ignore
                child.make_occupied()

    def __occupy_node(self, value: Any) -> None:
        if self._occupancy_mode == OctreeNode.OCCUPANCY_MODE_ITEMS:
            self.value = [value]
        else:
            self.value = True

    def check_occupancy(self) -> bool:
        """Checks if this node is occupied or not. Recursively checks all children.
        That is, for node to be occupied, any child must be occupied.

        Returns:
            bool: True if this node is occupied, False otherwise.
        """
        if not self.is_leaf():
            self.value = any([child.check_occupancy() for child in self._children])  # type: ignore
        return bool(self.value)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Gets the bounds of this node. Bounds are returned as minimum XYZ and maximum XYZ positions.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of arrays containing the minimum XYZ and maximum XYZ positions.
        """
        min_bounds = self._center - self._inner_radius
        max_bounds = self._center + self._inner_radius
        return min_bounds, max_bounds

    def gather_leaves(self) -> list['OctreeNode']:
        """Recursively finds all leaves in the subtree below this node and returns them in a list.
        If this node is a leaf, returns itself.

        Returns:
            list[OctreeNode]: List of leaves below and at the current node.
        """
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
        """Recursively finds all nodes in the subtree below this node and returns them in a dictionary.
        The keys are the zorder of the nodes and the values are the nodes themselves. Look up Z-order used in graphics.
        Basically, each node has a number (its own z-order) and its children have z-order starting with that number
        and after that number the child number is appended. E.g., z-order of node can be 431 and its children
        will have z-orders 4310 to 4317.

        Returns:
            dict[int, OctreeNode]: Dictionary of nodes in the subtree below this node.
        """
        if self.is_terminal():
            return {}
        else:
            result = self._children_dict
            for child in self._children:  # type: ignore # Check for terminal node above would catch empty children
                result.update(child.gather_zorder_dict())
            return result

    def __getitem__(self, key):
        """Children of node can be accessed using z-order (with or without prefix).
        For example, if node has z-order 431, child 2 can be accessed using 4312 or just 2.

        Args:
            key (int): Z-order of the child node to be accessed.

        Returns:
            OctreeNode: Child node if it exists, None otherwise. I.e., None means that child with the given index does not exist.
        """
        if key in self._children_dict:
            return self._children_dict[key]
        else:
            new_key = self.zorder * 10 + key
            if new_key in self._children_dict:
                return self._children_dict[new_key]
            else:
                return None

    def put(self, item: Any) -> bool:
        """For occupancy mode 'items' - adds an item to this node. Returns True if item was added.
        False means the ite is already in this node and thus put is not called recursively on the children.

        Args:
            item (Any): Item to be added to this node.

        Returns:
            bool: True if item was added (recursively to all children), False otherwise.
        """
        assert self._occupancy_mode == OctreeNode.OCCUPANCY_MODE_ITEMS, "Adding items to node is only allowed in 'items' mode!"
        assert self.is_leaf(), "Cannot add items to non-leaf nodes!"
        if item in self.value:
            return False
        self.__recursive_put(item)
        return True

    def __recursive_put(self, item: Any):
        self.__internal_put(item)
        if self.parent is not None:
            self.parent.__recursive_put(item)

    def __internal_put(self, item: Any):
        if self._occupancy_mode == OctreeNode.OCCUPANCY_MODE_ITEMS:
            self._value.append(item)  # type: ignore
        else:
            self._value = item
        return True

    def take(self, item: Any) -> bool:
        """Removes an item from this node. Returns True if item was removed, that is, it was contained in this node
        and this function was recursively called on all children. False means the item is not in this node and thus
        take is not called recursively on the children.
        Can only be used in 'items' occupancy mode.

        Args:
            item (Any): Item to be removed from this node.

        Returns:
            bool: True if item was removed (recursively to all children), False otherwise.
        """
        assert self._occupancy_mode == OctreeNode.OCCUPANCY_MODE_ITEMS, "Taking items from node is only allowed in 'items' mode!"
        assert self.is_leaf(), "Cannot take items from non-leaf nodes!"
        if item not in self.value:
            return False
        self.__recursive_take(item)
        return True

    def __recursive_take(self, item: Any):
        self.__internal_take(item)
        if self.parent is not None:
            self.parent.__recursive_take(item)

    def __internal_take(self, item: Any):
        self._value.remove(item)  # type: ignore # This function is only called in 'items' mode
        return True

    @property
    def full(self):
        # FIXME: should be more complex
        return bool(self.value)

    @property
    def empty(self):
        return not self.full

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
        """Inner radius is half of the side of the virtual cube represented by this node (voxel).
        It represents the distance between the center of the node and any side of the cube. That is,
        adding or subtracting inner radius from the center of the node will return coordinates
        of the sides/bounds of the cube/node.
        The name radius is because it is the radius of the sphere drawn inside the cube.

        Returns:
            float: Inner radius of this node.
        """
        return self._inner_radius

    @property
    def outer_radius(self):
        """Outer radius is half of the distance between the center of the node and any corner of the cube
        represented by this node (voxel). It is useful for a fast check if a point may be potentially inside
        the cube represented by this node. If the distance is greater than the outer radius, the point is
        outside the cube. If the distance is less than the outer radius, further check agains the inner radius
        is required.

        Returns:
            float: Outer radius of this node.
        """
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
        """Checks whether the given point lies withing the bounds of this node/voxel.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point lies within the bounds of this node, False otherwise.
        """
        if self.point_possibly_near(point) and self.point_within(point):
            return True
        else:
            return False

    def contains_point_recursive(self, point: np.ndarray) -> bool:
        """Checks whether the given point is included within a leaf in this sub-tree.
        Returns true only if there is a leaf in the sub-tree that contains the point.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point lies within a leaf in this sub-tree, False otherwise.
        """
        if self.contains_point(point):
            if self.is_leaf():
                return True
            if self.is_terminal():  # only count points if they are within nodes
                # TODO: expand non-leaf nodes
                return False
            for child in self._children:  # type: ignore # is_terminal would catch empty children
                if child.contains_point_recursive(point):
                    return True
            return False
        else:
            return False

    def contains_sphere(self, center: np.ndarray, radius: float) -> bool:
        # TODO: check if any corner lies within the sphere
        return self._sphere_in_sphere_sq(self.center, center, self._outer_radius_sq, radius * radius)

    def overlapped_by_sphere(self, center: np.ndarray, radius: float) -> bool:
        return self._point_in_sphere_sq(self.center, center, radius * radius)

    @staticmethod
    def _distance_sq(a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2

    def find_closest_leaf(self, point: np.ndarray, even_if_does_not_exist: bool = False) -> Optional['OctreeNode']:
        if even_if_does_not_exist:
            raise NotImplementedError("find_closest_leaf with even_if_does_not_exist=True is not implemented")
        if self.is_leaf():
            return self
        if self._children:
            closest_child = self._children[0]
            if len(self._children) > 1:
                closest_distance = self._distance_sq(closest_child.center, point)
                for child in self._children[1:]:
                    distance = self._distance_sq(child.center, point)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_child = child
            return closest_child.find_closest_leaf(point)
        return None

    def find_containing_leaf(self, point: np.ndarray, even_if_does_not_exist: bool = False) -> Optional['OctreeNode']:
        """Finds the leaf containing the specified point. Returns None if the leaf cannot be found for some reason
        (read below).
        If even_if_does_not_exist is False, it will only search existing leaves. That is, it will return None immediately
        if the containing non-leaf terminal node has no children or if it does not have child at the region of the point.
        If even_if_does_not_exist is False, this method will try to create virtual nodes if necessary.
        It will try to finds the containing leaf or "virtual" leaf that would be at that location. In this case,
        this method will return None only if the point is outside of the bounding box of the whole octree.

        Returns:
            Optional[OctreeNode]: Containing leaf node or None if it does not exist.
        """
        if even_if_does_not_exist:
            raise NotImplementedError("find_closest_leaf with even_if_does_not_exist=True is not implemented")
            # return self.__find_containing_leaf__(point)
        return self.__find_containing_leaf_naive__(point)

    def __find_containing_leaf_naive__(self, point: np.ndarray) -> Optional['OctreeNode']:
        if self.contains_point(point):
            if self.is_leaf():
                return self
            if self._children is not None:
                for child in self._children:
                    res = child.__find_containing_leaf_naive__(point)
                    if res is not None:
                        return res
        return None

    def __find_containing_leaf__(self, point: np.ndarray) -> Optional['OctreeNode']:
        if self.contains_point(point):
            if self.is_leaf():
                return self
            if self._children is not None:
                # TODO: add virtual nodes if necessary
                result_children = [child.__find_containing_leaf__(point) for child in self._children]
                if any(result_children):
                    return next((c for c in result_children if c is not None))
        return None

    def find_closest_leaf_to_bbox(self, bbox: SampledBBox, even_if_does_not_exist: bool = False) -> Optional['OctreeNode']:
        if even_if_does_not_exist:
            raise NotImplementedError("find_closest_leaf_to_aabb with even_if_does_not_exist=True is not implemented")
        if self.is_leaf():
            return self
        if self._children:
            closest_child = self._children[0]
            if len(self._children) > 1:
                bb_points = bbox.sample_edges_by_distance(self._inner_radius / 2).T
                closest_distance = np.min(self._distance_sq(closest_child.center, bb_points))
                for child in self._children[1:]:
                    distance = np.min(self._distance_sq(child.center, bb_points))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_child = child
            return closest_child.find_closest_leaf_to_bbox(bbox)
        return None

    def intersects_bbox(self, bbox: SampledBBox, even_if_does_not_exist: bool = False) -> bool:
        if even_if_does_not_exist:
            raise NotImplementedError("find_closest_leaf_to_aabb with even_if_does_not_exist=True is not implemented")
        # if voxel inside bbox -> True
        # voxel outside bbox -> False
        if self.is_leaf():
            pass
        if self._children:
            closest_child = self._children[0]
            if len(self._children) > 1:
                bb_points = bbox.sample_edges_by_distance(self._inner_radius / 2).T
                closest_distance = np.min(self._distance_sq(closest_child.center, bb_points))
                for child in self._children[1:]:
                    distance = np.min(self._distance_sq(child.center, bb_points))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_child = child
            return closest_child.find_closest_leaf_to_bbox(bbox)
        return None

    def expand_non_leaf(self) -> None:
        if self.is_leaf():
            return
        # TODO:

    def point_within(self, point: np.ndarray) -> bool:
        return self._point_in_aa_box(point, self._center, self._inner_radius)

    def point_possibly_near(self, point: np.ndarray) -> bool:
        return self._point_in_sphere_sq(point, self._center, self._outer_radius_sq)

    def add_node_at_point(self, point: np.ndarray) -> bool:
        """Adds a node at a given point or makes the closest leaf node occupied if it exists.
        This recursively tries to find the closest leaf. If at any point in the recursion there is no
        child node in the region where the point is located, it creates a new node. Subsequently,
        it will create all child nodes until leaf depth is reached. There, the node will be occupied.

        Args:
            point (np.ndarray): Point in space where the leaf should be added or value of leaf should be changed to occupied.

        Returns:
            bool: True if the node was added, False otherwise.
        """
        # TODO: Add node at point
        # if self.is_leaf():
        #     self._value = True
        #     return True
        # else:
        #     return self.find_closest_leaf(point).add_node_at_point(point)

    @staticmethod
    def _point_in_aa_box(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
        off_center_diff = np.abs(point - center)
        return all(off_center_diff < radius)

    @staticmethod
    def _point_in_any_box(point: np.ndarray, o: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        # OP→⋅OA→,OP→⋅OB→,OP→⋅OC→,AP→⋅AO→,BP→⋅BO→,CP→⋅CO >= 0
        return np.logical_and(
            np.logical_and(
                np.logical_and(
                    np.dot(point - o, a - o) >= 0,
                    np.dot(point - o, b - o) >= 0,
                ),
                np.logical_and(
                    np.dot(point - o, c - o) >= 0,
                    np.dot(point - a, o - a) >= 0,
                )
            ),
            np.logical_and(
                np.dot(point - b, o - b) >= 0,
                np.dot(point - c, o - c) >= 0
            )
        )

    @staticmethod
    def _point_in_sphere_sq(point: np.ndarray, center: np.ndarray, radius_squared: float) -> bool:
        """
        Check if a point is inside a sphere defined by its center and squared radius.

        :param point: A numpy array representing the point coordinates.
        :param center: A numpy array representing the center of the sphere.
        :param radius_squared: The squared radius of the sphere.
        :return: True if the point is inside the sphere, False otherwise.
        """
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
        inner_radius_child = (leaf_scale * (2 ** (depth - 1))) / 2

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
