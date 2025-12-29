"""Module for prior distributions."""

from typing import Literal

import numpy

type BOUNDARY_TYPES = None | Literal["periodic", "reflective"]


class Prior:
    """
    Prior base class.

    Parameters
    ----------
    boundary
        Boundary condition for the prior.
    """

    def __init__(self, boundary: BOUNDARY_TYPES = None) -> None:
        is_periodic = is_reflective = False
        if boundary == "periodic":
            is_periodic = True
        elif boundary == "reflective":
            is_reflective = True
        self.is_periodic = is_periodic
        self.is_reflective = is_reflective

    def transform(self, u: float) -> float:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        msg = "Priors must implement this method."
        raise NotImplementedError(msg)


class DeltaFunction(Prior):
    """
    Delta function prior.

    Parameters
    ----------
    x_peak
        Location of the delta function peak.
    """

    def __init__(self, x_peak: float) -> None:
        boundary = None
        super().__init__(boundary)
        self.x_peak = x_peak

    def transform(self, u: float) -> float:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        return self.x_peak


class Uniform(Prior):
    """
    Uniform prior.

    Parameters
    ----------
    x_min
        Minimum value of the prior.
    x_max
        Maximum value of the prior.
    boundary
        Boundary condition for the prior.
    """

    def __init__(self, x_min: float, x_max: float, boundary: BOUNDARY_TYPES = None) -> None:
        super().__init__(boundary)
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, u: float) -> float:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        return self.x_min + (self.x_max - self.x_min) * u


class Cosine(Prior):
    """
    Cosine prior.

    Parameters
    ----------
    x_min
        Minimum value of the prior.
    x_max
        Maximum value of the prior.
    boundary
        Boundary condition for the prior.
    """

    def __init__(
        self, x_min: float = -numpy.pi / 2, x_max: float = numpy.pi / 2, boundary: BOUNDARY_TYPES = None
    ) -> None:
        super().__init__(boundary)
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, u: float) -> numpy.float64:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        sin_min = numpy.sin(self.x_min)
        sin_max = numpy.sin(self.x_max)
        x: numpy.float64 = numpy.arcsin(sin_min + (sin_max - sin_min) * u)
        return x


class Sine(Prior):
    """
    Sine prior.

    Parameters
    ----------
    x_min
        Minimum value of the prior.
    x_max
        Maximum value of the prior.
    boundary
        Boundary condition for the prior.
    """

    def __init__(self, x_min: float = 0, x_max: float = numpy.pi, boundary: BOUNDARY_TYPES = None) -> None:
        super().__init__(boundary)
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, u: float) -> numpy.float64:
        """
        Transform sample between 0 and 1 to prior space.

        Parameters
        ----------
        u
            Random sample between 0 and 1.

        Returns
        -------
        x
            Transformed sample in prior space.
        """
        cos_min = numpy.cos(self.x_min)
        cos_max = numpy.cos(self.x_max)
        x: numpy.float64 = numpy.arccos(cos_min + (cos_max - cos_min) * u)
        return x


class Priors(list[Prior]):
    """
    Collection of prior distributions.

    Parameters
    ----------
    priors
        List of prior distributions.
    """

    def __init__(self, priors: list[Prior]) -> None:
        super().__init__(priors)

    def transform(self, u: numpy.typing.NDArray[numpy.floating]) -> numpy.typing.NDArray[numpy.floating]:
        """
        Transform unit hypercube samples to prior space.

        Parameters
        ----------
        u
            Random samples from the unit hypercube.

        Returns
        -------
        x
            Transformed samples in prior space.
        """
        x = u.copy()
        for i, prior in enumerate(self):
            x[i] = prior.transform(u[i])
        return x

    @property
    def periodic_indices(self) -> None | list[int]:
        """Indices of periodic parameters."""
        return [i for i, prior in enumerate(self) if prior.is_periodic] or None

    @property
    def reflective_indices(self) -> None | list[int]:
        """Indices of reflective parameters."""
        return [i for i, prior in enumerate(self) if prior.is_reflective] or None
