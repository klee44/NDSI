## Overview
Tested KANs as a parameterization technique for neural ODEs

All implemented in PyTorch

Required packages:
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

- [efficient kan](https://github.com/Blealtan/efficient-kan) (included in the repo)


### to run the experiments on ODEBench (from ODEFormer)

```python
python train_kan.py --kanlayer 3 2 --grid 5 --k 3 
```

- use `kanlayer` to specify the kan architecture, 

- use `grid` to specify the grid on which B-splines are defined 

- use `k` to specify the degree of B-splines
