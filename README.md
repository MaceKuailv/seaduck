# seaduck

A python package that interpolates data from ocean dataset from both Eulerian and Lagrangian perspective.

## Quick Start

```python
>>> import seaduck

```

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. seaduck):

```
conda create -n seaduck
conda activate seaduck
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`
1. Build the documentation: `make docs-build`

## License

```
Copyright 2023, Wenrui Jiang.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
