# Copyright 2023, Wenrui Jiang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    # NOTE: the `version.py` file must not be present in the git repository
    #   as it is generated by setuptools at install time
    from seaduck.version import __version__
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "999"

from seaduck.eulerian import Position
from seaduck.kernel_weight import KnW
from seaduck.lagrangian import Particle
from seaduck.ocedata import OceData
from seaduck.oceinterp import OceInterp
from seaduck.runtime_conf import rcParam
from seaduck.topology import Topology

__all__ = [
    "__version__",
    "Position",
    "KnW",
    "Particle",
    "OceData",
    "OceInterp",
    "rcParam",
    "Topology",
]
