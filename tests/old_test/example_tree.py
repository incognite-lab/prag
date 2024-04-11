import numpy as np
import pytest

from octree.octree_root import OctreeRoot
from octree.sampled_boxes import SampledAABB, SampledBBox


@pytest.fixture
def octree_root_instance():
    root_depth = 3
    leaf_scale = 0.1
    center = np.array([0, 0, 0])
    return OctreeRoot.create_dense(root_depth, leaf_scale, center)


def test_process_occupancy_grid(octree_root_instance, benchmark):
    mat = np.zeros((3, 3, 3), dtype=np.uint8)
    result = benchmark(OctreeRoot.create_from_occupancy_tensor, mat, 0.1, np.array([0, 0, 0]), 1)
    assert result is not None


def test_initialize_as_dense_tree(octree_root_instance, benchmark):
    result = benchmark(OctreeRoot.create_dense, 3, 0.1, np.array([0, 0, 0]))
    assert result is not None


def test_make_leaf_empty_or_occupied(octree_root_instance):
    leaf = octree_root_instance.find_closest_leaf(np.array([0.1, 0.1, 0.1]))
    leaf_empty = leaf.value
    leaf.set_value(not leaf_empty)
    assert leaf.value != leaf_empty


def test_find_closest_leaf(octree_root_instance):
    closest_leaf = octree_root_instance.find_closest_leaf(np.array([0.1, 0.1, 0.1]))
    assert closest_leaf is not None


def test_check_point_contained_in_tree(octree_root_instance):
    point = np.array([0.1, 0.1, 0.1])
    assert octree_root_instance.contains_point(point)


def test_invalid_inputs_for_functions(octree_root_instance):
    with pytest.raises(ValueError):
        pass
    # Add more scenarios with incorrect inputs


if __name__ == "__main__":
    a, b = np.random.random(3) * 5, np.random.random(3) * 5
    bb = SampledAABB(a, b)

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

    node = octree.find_closest_leaf_to_aabb(bb)
    import open3d as o3d
    tree_bbox = o3d.geometry.AxisAlignedBoundingBox(*octree.root.get_bounds())
    tree_bbox.color = np.r_[1, 0, 0]
    mesh_bbox = o3d.geometry.AxisAlignedBoundingBox(bb._min, bb._max)
    mesh_bbox.color = np.r_[0, 0, 1]

    bbs = [tree_bbox, mesh_bbox]
    for n in octree.leaves:
        bb = o3d.geometry.AxisAlignedBoundingBox(*n.get_bounds())
        bb.color = np.r_[0.1, 0.1, 0.1]
        bbs.append(bb)
    bb = o3d.geometry.AxisAlignedBoundingBox(*node.get_bounds())
    bb.color = np.r_[0, 1, 0]
    bbs.append(bb)
    o3d.visualization.draw_geometries(bbs)

    # print(octree.print_stats())
    # # print(octree.pretty_print())
    # drawer = OTDrawer(octree.root, exclude_empty=not draw_empty)
    # # print(list(drawer.graph.nodes(data=True)))

    # drawer.render()

