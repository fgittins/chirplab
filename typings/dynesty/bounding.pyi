from typing import Any, Literal, overload

import numpy

class Bound:
    def __init__(self, ndim: int) -> None: ...
    def contains(self, x: numpy.typing.NDArray[numpy.floating]) -> bool: ...
    def sample(self, rstate: None | numpy.random.Generator = None) -> numpy.typing.NDArray[numpy.floating]: ...
    def samples(
        self, nsamples: int, rstate: None | numpy.random.Generator = None
    ) -> numpy.typing.NDArray[numpy.floating]: ...
    def get_random_axes(self, rstate: numpy.random.Generator) -> numpy.typing.NDArray[numpy.integer]: ...
    def scale_to_logvol(self, logvol: float) -> None: ...
    def update(
        self,
        points: numpy.typing.NDArray[numpy.floating],
        rstate: None | numpy.random.Generator = None,
        bootstrap: int = 0,
        pool: Any = None,
    ) -> None: ...

class UnitCube(Bound): ...

class Ellipsoid(Bound):
    def __init__(
        self,
        ndim: int,
        ctr: None | numpy.typing.NDArray[numpy.floating] = None,
        cov: None | numpy.typing.NDArray[numpy.floating] = None,
        am: None | numpy.typing.NDArray[numpy.floating] = None,
        axes: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> None: ...
    def major_axis_endpoints(
        self,
    ) -> tuple[numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.floating]]: ...
    def distance(self, x: numpy.typing.NDArray[numpy.floating]) -> float: ...
    def distance_many(self, x: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]: ...
    def unitcube_overlap(self, ndraws: int = 10_000, rstate: None | numpy.random.Generator = None) -> float: ...
    def update(
        self,
        points: numpy.typing.NDArray[numpy.floating],
        rstate: None | numpy.random.Generator = None,
        bootstrap: int = 0,
        pool: Any = None,
        mc_integrate: bool = False,
    ) -> None: ...

class MultiEllipsoid(Bound):
    def __init__(
        self,
        ndim: int,
        ells: None | list[Ellipsoid] = None,
        ctrs: None | list[numpy.typing.NDArray[numpy.floating]] = None,
        covs: None | list[numpy.typing.NDArray[numpy.floating]] = None,
    ) -> None: ...
    def within(
        self, x: numpy.typing.NDArray[numpy.floating], j: None | int = None
    ) -> numpy.typing.NDArray[numpy.integer]: ...
    def overlap(self, x: numpy.typing.NDArray[numpy.floating]) -> int: ...
    @overload  # type: ignore[override]
    def sample(
        self, rstate: None | numpy.random.Generator = None, return_q: Literal[False] = False
    ) -> tuple[numpy.typing.NDArray[numpy.floating], int]: ...
    @overload
    def sample(
        self, rstate: None | numpy.random.Generator = None, return_q: Literal[True] = True
    ) -> tuple[numpy.typing.NDArray[numpy.floating], int, int]: ...
    @overload
    def monte_carlo_logvol(
        self, ndraws: int = 10_000, rstate: None | numpy.random.Generator = None, return_overlap: Literal[True] = True
    ) -> tuple[float, float]: ...
    @overload
    def monte_carlo_logvol(
        self, ndraws: int = 10_000, rstate: None | numpy.random.Generator = None, return_overlap: Literal[False] = False
    ) -> float: ...
    def update(
        self,
        points: numpy.typing.NDArray[numpy.floating],
        rstate: None | numpy.random.Generator = None,
        bootstrap: int = 0,
        pool: Any = None,
        mc_integrate: bool = False,
    ) -> None: ...

class RadFriends(Bound):
    def __init__(self, ndim: int, cov: None | numpy.typing.NDArray[numpy.floating] = None) -> None: ...
    def within(
        self, x: numpy.typing.NDArray[numpy.floating], j: None | int = None
    ) -> numpy.typing.NDArray[numpy.integer]: ...
    def overlap(self, x: numpy.typing.NDArray[numpy.floating]) -> int: ...
    @overload
    def sample(
        self, rstate: None | numpy.random.Generator = None, return_q: Literal[False] = False
    ) -> numpy.typing.NDArray[numpy.floating]: ...
    @overload
    def sample(
        self, rstate: None | numpy.random.Generator = None, return_q: Literal[True] = True
    ) -> tuple[numpy.typing.NDArray[numpy.floating], int]: ...
    @overload
    def monte_carlo_logvol(
        self, ndraws: int = 10_000, rstate: None | numpy.random.Generator = None, return_overlap: Literal[True] = True
    ) -> tuple[float, float]: ...
    @overload
    def monte_carlo_logvol(
        self, ndraws: int = 10_000, rstate: None | numpy.random.Generator = None, return_overlap: Literal[False] = False
    ) -> float: ...
    def update(
        self,
        points: numpy.typing.NDArray[numpy.floating],
        rstate: None | numpy.random.Generator = None,
        bootstrap: int = 0,
        pool: Any = None,
        mc_integrate: bool = False,
        use_clustering: bool = True,
    ) -> None: ...

class SupFriends(Bound):
    def __init__(self, ndim: int, cov: None | numpy.typing.NDArray[numpy.floating] = None) -> None: ...
    def within(
        self, x: numpy.typing.NDArray[numpy.floating], j: None | int = None
    ) -> numpy.typing.NDArray[numpy.integer]: ...
    def overlap(self, x: numpy.typing.NDArray[numpy.floating]) -> int: ...
    @overload
    def sample(
        self, rstate: None | numpy.random.Generator = None, return_q: Literal[False] = False
    ) -> numpy.typing.NDArray[numpy.floating]: ...
    @overload
    def sample(
        self, rstate: None | numpy.random.Generator = None, return_q: Literal[True] = True
    ) -> tuple[numpy.typing.NDArray[numpy.floating], int]: ...
    @overload
    def monte_carlo_logvol(
        self, ndraws: int = 10_000, rstate: None | numpy.random.Generator = None, return_overlap: Literal[True] = True
    ) -> tuple[float, float]: ...
    @overload
    def monte_carlo_logvol(
        self, ndraws: int = 10_000, rstate: None | numpy.random.Generator = None, return_overlap: Literal[False] = False
    ) -> float: ...
    def update(
        self,
        points: numpy.typing.NDArray[numpy.floating],
        rstate: None | numpy.random.Generator = None,
        bootstrap: int = 0,
        pool: Any = None,
        mc_integrate: bool = False,
        use_clustering: bool = True,
    ) -> None: ...
