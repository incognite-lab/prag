from typing import Union

import networkx as nx
import numpy as np
from mayavi import mlab

from octree.octree_node import OctreeNode
from octree.octree_root import OctreeRoot

nx_node_type = tuple[int, int]


class OTDrawer():

    def __init__(self, tree_root: Union[OctreeNode, OctreeRoot], exclude_empty: bool = False) -> None:
        if isinstance(tree_root, OctreeRoot):
            tree_root = tree_root.root
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
        tube = mlab.pipeline.tube(pts, tube_radius=0.0005)
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
