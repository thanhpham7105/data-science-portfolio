# Optimization / Linear Programming

## Overview
Three linear/mixed-integer programming projects applying mathematical optimization (via PuLP/CBC) to logistics and network-flow business problems.

## 1. Business case analysis
Identified a weekly delivery-distribution problem -- matching truck capacity to delivery demand while minimizing cost -- as a linear programming opportunity, and made the case for an optimization approach over ad hoc planning.

## 2. Solving a network-flow optimization problem
Formulated and solved a multi-hub freight network-flow problem (hubs -> transfer points -> destination centers) to minimize total transportation cost subject to capacity and demand constraints.

- `network_flow_optimization.py` -- the full PuLP model: decision variables for hub-to-transfer, hub-to-destination, and transfer-to-destination shipments; an objective function minimizing total transportation cost; hub-capacity, flow-balance, and destination-demand constraints; solved with the CBC solver.

## 3. Constraint verification
Verified the solver's optimal solution against all hub-capacity, flow-balance, demand, and non-negativity constraints, confirming a feasible and cost-minimizing routing plan.

## Skills demonstrated
Linear/mixed-integer programming formulation, PuLP/CBC solver usage, constraint verification, translating optimization output into operational recommendations.
