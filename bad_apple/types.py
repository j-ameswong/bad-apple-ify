from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

type Image = npt.NDArray[np.uint8]

type Brightness = npt.NDArray[np.float64]

type Indices = npt.NDArray[np.int64]

type Fit = Literal["native", "crop", "stretch"]

type MetricName = Literal["brightness", "colour"]
