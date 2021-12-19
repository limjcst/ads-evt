# Anomaly Detection in Streams with Extreme Value Theory

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Actions](https://github.com/limjcst/ads-evt/workflows/Actions/badge.svg)](https://github.com/limjcst/ads-evt/actions)

This repository wraps the original implementation of [SPOT published in KDD'17](https://github.com/Amossys-team/SPOT) as an installable package.
We refactor the original one, removing duplicated code.
To verify the faithfulness, several test cases are introduced in `tests/test_faithfulness.py`.

## Usage

Install this package via `pip install ads-evt`.

`ads_evt` has almost the same interface as the original implementation.

```python
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import ads_evt as spot


# physics.dat is a file in the original repository
with open("physics.dat", encoding="UTF-8") as obj:
    data = np.array(list(map(float, obj.read().split(","))))
init_data = 2000
proba = 1e-3
depth = 450

models: List[spot.SPOTBase] = [
    # spot.SPOT(q=proba),
    # spot.dSPOT(q=proba, depth=depth),
    # spot.biSPOT(q=proba),
    # The original implementation of bidSPOT uses n_points=8 for _grimshaw by default
    spot.bidSPOT(q=proba, depth=depth, n_points=8),
]
for alg in models:
    alg.fit(init_data=init_data, data=data)
    alg.initialize()
    results = alg.run()
    # Plot
    figs = alg.plot(results)
    plt.show()
```

### For developers

Execute test cases with the following commands

```bash
# Install dependencies for development
git submodule update --init
python -m pip install -r requirements-dev.txt
# Execute test cases
coverage run
coverage report
```

## Licences

GNU GPLv3
