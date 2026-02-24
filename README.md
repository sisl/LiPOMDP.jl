# LitPOMDP

[![CI](https://github.com/sisl/LitPOMDP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/LitPOMDP.jl/actions/workflows/CI.yml)

A Julia package for modeling lithium mining operations as a Partially Observable Markov Decision Process (POMDP). The framework simulates decision-making under uncertainty for lithium extraction using both Direct Lithium Extraction (DLE) and traditional mining methods.

## Features

- **Multi-site mining decisions**: Model considers 4 deposit sites (2 DLE, 2 traditional)
- **Automatic mine operation**: Once opened, mines produce automatically without explicit operate actions
- **Environmental considerations**: Tracks CO2 emissions with different costs for DLE vs traditional mining
- **Market dynamics**: Multiple pricing models (static, linear, exponential, GBM, historical)
- **Uncertainty handling**: Manages uncertainty in deposit sizes, prices, and demand
- **Policy optimization**: Uses POMCPOW for finding optimal mining strategies

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sisl/LitPOMDP.jl")
```

For development:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

```julia
using LitPOMDP
using POMDPs
using POMCPOW

# Create POMDP instance
pomdp = LiPOMDP(
    price_model_type=1,  # Static pricing (default: 1)
    time_horizon=29,      # 29-year planning horizon (default: 29)
    alpha=0.5             # Balance between profit and emissions (default: 0.5)
)

# Setup initial belief
s0, initial_belief = create_initial_belief(pomdp)

# Create POMCPOW solver
solver = POMCPOWSolver(
    tree_queries=1000,
    max_depth=10,
    criterion=MaxUCB(20.0)
)

policy = solve(solver, pomdp)
```

## State and Action Structure

### State
```julia
@with_kw mutable struct State
    deposit::Vector{Float64}      # Remaining deposits at each site (in-situ Li in tonnes)
    time::Float64                 # Current time step (year)
    price::Float64                # Current market price ($/tonne LCE)
    company_demand::Float64       # Company's target demand (tonnes LCE/year)
    total_mined::Float64          # Cumulative sold LCE (tonnes)
    have_opened::Vector{Bool}    # Which mines are opened (operate automatically)
    is_depleted::Vector{Bool}     # Which mines are depleted
    restored_mines::Vector{Bool}  # Which mines have been restored
end
```

### Action
```julia
@with_kw mutable struct Action
    site::Vector{Int64}   # Sites to act on (0=DoNothing, 1-4=site number)
    type::Vector{Int64}   # Action types (0=DoNothing, 1=explore, 2=open, 3=restore)
end
```

Note: Mines operate automatically once opened - no explicit "operate" action needed. Depleted mines can be restored using action type 3.

## Available Policies

The package includes several heuristic policies for comparison:

- **ProfitMaximizerPolicy**: Maximizes expected profit by selecting the most profitable mine to open
- **EmissionMinimizerPolicy**: Minimizes emissions by selecting mines with lowest emission costs
- **ExploreOnlyPolicy**: Prioritizes exploration to gather information before opening mines
- **RandomHeuristicPolicy**: Random baseline for comparison
- **DynamicProfitMaximizerPolicy**: Dynamically evaluates and selects the most profitable action at each step
- **DynamicEmissionMinimizerPolicy**: Dynamically evaluates and selects actions to minimize emissions
- **OneStepLookaheadPolicy**: Uses one-step lookahead with rollouts to evaluate actions
- **AlphaAwareRolloutPolicy**: Rollout policy that balances profit and emissions based on alpha parameter

## Running Experiments

The package includes several scripts for running experiments and generating visualizations:

- **`scripts/running_policy_tests.jl`**: Comprehensive policy comparison across multiple price models and alpha values
- **`scripts/simple_policy_test.jl`**: Simplified policy testing framework
- **`scripts/final_results_simulation.jl`**: Full simulation with train/test split for robust evaluation
- **`scripts/price_model_comparison.jl`**: Compare policies across different price model combinations
- **`scripts/plot_belief_uncertainty.jl`**: Generate plots showing belief uncertainty evolution over time
- **`scripts/policy_belief_action_timeline.jl`**: Create timeline visualizations of beliefs and actions

These scripts provide:
- Mean rewards and discounted rewards for each policy
- Statistical analysis (std dev, min/max, median)
- Profit vs. emissions Pareto curves
- Belief uncertainty evolution over time
- Action timeline comparisons

## Pricing Models

The package supports multiple pricing models:
- **Type 1**: Static pricing - constant price with small random variations
- **Type 2**: Linear pricing - price increases linearly over time
- **Type 3**: Exponential pricing - price grows exponentially over time
- **Type 4**: GBM (Geometric Brownian Motion) - stochastic price evolution with drift and volatility
- **Type 6**: Historical pricing - uses actual historical lithium price data from 1995-2024

