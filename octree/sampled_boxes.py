import numpy as np


class SampledBBox:

    @classmethod
    def from_o3d_obb(cls, o3d_obb) -> 'SampledBBox':
        corners = np.asarray(o3d_obb.get_box_points())
        cornermap = [
            0, 1, 2, 7, 3, 6, 5, 4
        ]
        return cls(corners[cornermap])

    def __init__(self, corners) -> None:
        self._edge_indices = [
            (0, 1),
            (1, 5),
            (5, 4),
            (4, 0),
            (2, 3),
            (3, 7),
            (7, 6),
            (6, 2),
            (0, 2),
            (1, 3),
            (5, 7),
            (4, 6)
        ]
        self._corners = corners
        self.o = self.corners[0]
        self.a = self.corners[1]
        self.b = self.corners[2]
        self.c = self.corners[4]
        self.oa = self.corners[1] - self.corners[0]
        self.ob = self.corners[2] - self.corners[0]
        self.oc = self.corners[4] - self.corners[0]
        self.ao = self.corners[0] - self.corners[1]
        self.bo = self.corners[0] - self.corners[2]
        self.co = self.corners[0] - self.corners[4]

        # self._face_normals = self._compute_face_normals()

    def sample_edges(self, N) -> np.ndarray:
        # Sample points along each edge of the bounding box
        sampled_points = np.vstack([
            self.sample_points_on_line_segment(self._corners[i1], self._corners[i2], N)
            for i1, i2 in self._edge_indices
        ])
        return sampled_points

    def sample_edges_by_distance(self, distance) -> np.ndarray:
        # Sample points along each edge of the bounding box
        sampled_points = []
        for i1, i2 in self._edge_indices:
            c1, c2 = self._corners[i1], self._corners[i2]
            n = max(2, int(np.ceil(np.linalg.norm(c2 - c1) / distance)))
            sampled_points.append(self.sample_points_on_line_segment(c1, c2, n))

        sampled_points = np.vstack(sampled_points)
        return sampled_points

    @staticmethod
    def sample_points_on_line_segment(point1, point2, N):
        assert N > 1, "N must be greater than 2 (at least the start and end points)!"
        sampled_points = np.column_stack((
            np.linspace(point1[0], point2[0], N),
            np.linspace(point1[1], point2[1], N),
            np.linspace(point1[2], point2[2], N)
        ))
        return sampled_points

    def contains_point(self, point: np.ndarray) -> bool:
        # return all(self._corners.min(axis=0) <= point) and all(point <= self._corners.max(axis=0))
        # OP→⋅OA→,OP→⋅OB→,OP→⋅OC→,AP→⋅AO→,BP→⋅BO→,CP→⋅CO >= 0
        return np.logical_and(
            np.logical_and(
                np.logical_and(
                    np.dot(point - self.o, self.oa) >= 0,
                    np.dot(point - self.o, self.ob) >= 0,
                ),
                np.logical_and(
                    np.dot(point - self.o, self.oc) >= 0,
                    np.dot(point - self.a, self.ao) >= 0,
                )
            ),
            np.logical_and(
                np.dot(point - self.b, self.bo) >= 0,
                np.dot(point - self.c, self.co) >= 0
            )
        )

    @property
    def corners(self) -> np.ndarray:
        return self._corners

    @property
    def edge_indices(self):  # -> list[tuple[int, int]]:
        return self._edge_indices

    @property
    def edges(self) -> np.ndarray:
        return self._corners[self._edge_indices]

    # @property
    # def face_normals(self) -> np.ndarray:
    #     return self._face_normals


class SampledAABB(SampledBBox):
    def __init__(self, min: np.ndarray, max: np.ndarray) -> None:
        self._min = min
        self._max = max
        super().__init__(self.__compute_corners(self._min, self._max))

    @staticmethod
    def __compute_corners(min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
        # Compute the 8 corners of the axis aligned bounding box
        #   6 ------ 7
        #  /|      /|
        # 2 4----- 3 5
        # |/      |/
        # 0 ------ 1

        # y  z
        # | /
        # o--x
        #
        corners = np.array([
            min_bounds,
            [max_bounds[0], min_bounds[1], min_bounds[2]],
            [min_bounds[0], max_bounds[1], min_bounds[2]],
            [max_bounds[0], max_bounds[1], min_bounds[2]],
            [min_bounds[0], min_bounds[1], max_bounds[2]],
            [max_bounds[0], min_bounds[1], max_bounds[2]],
            [min_bounds[0], max_bounds[1], max_bounds[2]],
            max_bounds
        ])
        return corners
