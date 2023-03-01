# petrinets

Python implementation of Petri nets.

## Usage

The following example is the SIR epidemic model, modeled by the reaction network

$$ S + I \to 2I $$

$$ I \to R $$

We can define the corresponding Petri net using the primary class `petrinet.PetriNet`:

```python
from petrinet import PetriNet
S, I, R = "SIR"
a, b = "ab"
net = PetriNet(
  species     = [S, I, R],
  transitions = [a, b],
  edges_in  = [(S,a), (I,a), (I,b)],
  edges_out = [(I,a), (I,a), (R,b)]
)
```

By assigning each reaction a rate, we can compute the time derivative using mass action kinetics:

```python
from mass_action import time_dot
time_dot(net, x=[1.0, 2.0, 3.0], rates=[1.0, 1.2])
```

Use ODE solvers such as scipy's solve_ivp:

```python
from scipy.integrate import solve_ivp

soln = solve_ivp(
  fun=lambda t, x: time_dot(net, x=x, rates=[2.0, 1.0]),
  t_span=[0.0, 10.0],
  y0=[0.99, 0.01, 0.00],
  method="RK45"
)
```

Plot the solution with matplotlib:

```python
import matplotlib.pyplot as plt
for i in range(len(soln.y)):
  plt.plot(soln.t, soln.y[i], label=i)
plt.legend()
plt.show()
```

![Time series plot of the SIR model](figures/ode_demo.png)
