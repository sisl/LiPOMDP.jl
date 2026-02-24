# Import without other commands doesn't do anything

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using LitPOMDP
using JSON
using POMCPOW: POMCPOWPlanner  # so `isa POMCPOWPlanner` works
using CSV
using DataFrames

# Removed tree generation functionality

"""
Price Model Comparison Script - Focus on alpha=0.5 with all price model combinations
"""

# Removed action_to_string function - not needed for this analysis

function create_policy(policy_type::String, pomdp::LiPOMDP)
    """Create a policy based on type."""
    
    if policy_type == "POMCPOW"
        solver = POMCPOWSolver(
            tree_queries = 5000, #give it time to plan
            max_depth = 29, #max time step you can get, planning over POMDP in the future
            criterion = MaxUCB(30.0),#increased exploration constant for high uncertainty deposits
            # Progressive widening (actions)
            enable_action_pw = true,
            k_action = 5.0,  # increased for more action exploration
            alpha_action = 0.5,  # increased for more progressive widening
            # Progressive widening (observations)
            k_observation = 3.0,  # increased for more observation exploration
            alpha_observation = 0.3,  # increased for more observation progressive widening
            estimate_value = 0.0, 
            tree_in_info = true,
           # rng = MersenneTwister(1234)
        )
        return solve(solver, pomdp)
        
    elseif policy_type == "ProfitMaximizer"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])
        
    elseif policy_type == "EmissionMinimizer"
        return EmissionMinimizerPolicy(pomdp, top_k=1, times=[1])
        
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp)
        
    elseif policy_type == "Random"
        return RandomHeuristicPolicy(pomdp, debug=false)
        
    elseif policy_type == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false)
        
    elseif policy_type == "DynamicEmissionMinimizer"
        return DynamicEmissionMinimizerPolicy(pomdp, debug=false)
        
    else
        error("Unknown policy type: $policy_type")
    end
end

function evaluate_policy_simple(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                              policy, updater::LiBeliefUpdater, n_sims::Int=50; 
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    """Evaluate a policy with key metrics."""
    
    total_rewards = Float64[]
    total_discounted_rewards = Float64[]
    total_emission_costs = Float64[]
    total_profits = Float64[]
    total_demands = Float64[]
    total_productions = Float64[]
    all_histories = []
    
    for i in 1:n_sims
        Random.seed!(i) #control randomness to be deterministic across runs for comparability
        #so initial belief and state are the same across runs -- be a deterministic belief that is still random
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_emission_cost = 0.0
        total_profit = 0.0
        total_demand = 0.0
        total_production = 0.0
        history = []
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Get action from policy
            a = action(policy, b)

            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)

            costs = compute_costs(evaluation_pomdp, s, a)  
            emissions = compute_emissions(evaluation_pomdp, s, a)  
            current_production, _ = calculate_total_production(evaluation_pomdp, s)
            
            revenue = current_production * s.price
            
            profit = revenue - costs    
            
            total_demand += s.company_demand
            total_production += current_production
            total_profit += profit
            total_emission_cost += emissions
            
            # Track rewards
            total_reward += r  # Keep original reward for comparison
            total_discounted_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end

            dict_step_result = Dict(
                "s" => s,
                "a" => a,
                "o" => o,
                "r" => r,
                "b" => b,
            )

            push!(history, dict_step_result)
            
            # Update belief using planning POMDP's updater
            b = update(updater, b, a, o)
            
            s = sp
        end
        
        # Store results
        push!(total_rewards, total_reward)
        push!(total_discounted_rewards, total_discounted_reward)
        push!(total_emission_costs, total_emission_cost)
        push!(total_profits, total_profit)
        push!(total_demands, total_demand)
        push!(total_productions, total_production)
        push!(all_histories, history)
    end
    
    return Dict(
        "total_reward" => total_rewards,
        "total_discounted_reward" => total_discounted_rewards,
        "total_emission_cost" => total_emission_costs,
        "total_profit" => total_profits,
        "total_demand" => total_demands,
        "total_production" => total_productions,
        "history" => all_histories
    )
end

function run_price_model_comparison()
    """Run price model comparison for alpha=0.5 with all combinations."""
    
    println("🚀 PRICE MODEL COMPARISON TESTING")
    println("="^80)
    println("Testing all price model combinations (1,2,3,6) for planning vs evaluation")
    println("Focus on alpha=0.5 (balanced profit/emission optimization)")
    println("="^80)
    
    # Configuration
    alpha = 0.5
    n_sims = 30
    time_horizon = 25
    policies = ["DynamicProfitMaximizer", "ExploreOnly", "ProfitMaximizer", "EmissionMinimizer", "DynamicEmissionMinimizer", "POMCPOW"]
    price_models = [1, 2, 3, 6]  # Static, Linear, Exponential, Historical
    
    println("Configuration:")
    println("  • Alpha: $alpha")
    println("  • Simulations per policy: $n_sims")
    println("  • Time horizon: $time_horizon")
    println("  • Price models: $price_models")
    println("  • Policies: $policies")
    
    # Create output directory
    output_dir = "outputs/price_model_comparison"
    isdir(output_dir) || mkpath(output_dir)
    
    # Test each price model combination
    total_configs = length(price_models) * length(price_models)
    config_count = 0
    
    for planning_model in price_models
        for eval_model in price_models
            config_count += 1
            
            println("\n[$(config_count)/$(total_configs)] Testing Planning Model $planning_model vs Evaluation Model $eval_model")
            
            # Create planning and evaluation POMDPs
            planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon, alpha=alpha)
            evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon, alpha=alpha)
            
            # Create belief updater
            updater = LiBeliefUpdater(P=evaluation_pomdp)
            
            # Store results for this combination
            results_data = []
            
            # Test each policy
            for policy_type in policies
                println("  Testing $policy_type...")
                
                # Create policy
                policy = create_policy(policy_type, planning_pomdp)
                
                # Evaluate policy
                results = evaluate_policy_simple(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                
                # Calculate statistics
                mean_profit = mean(results["total_profit"])
                std_profit = std(results["total_profit"])
                mean_emission_cost = mean(results["total_emission_cost"])
                std_emission_cost = std(results["total_emission_cost"])
                mean_disc_reward = mean(results["total_discounted_reward"])
                std_disc_reward = std(results["total_discounted_reward"])
                
                # Calculate standard error for discounted reward
                std_error_disc_reward = std_disc_reward / sqrt(n_sims)
                
                # Store results
                push!(results_data, (
                    policy_type,
                    mean_profit,
                    std_profit,
                    mean_emission_cost,
                    std_emission_cost,
                    mean_disc_reward,
                    std_disc_reward,
                    std_error_disc_reward
                ))
                
                println("    ✓ Profit: \$$(round(mean_profit/1e6, digits=2))M ± $(round(std_profit/1e6, digits=2))M")
                println("    ✓ Emissions: \$$(round(mean_emission_cost/1e6, digits=2))M ± $(round(std_emission_cost/1e6, digits=2))M")
                println("    ✓ Disc. Reward: $(round(mean_disc_reward, digits=2)) ± $(round(std_error_disc_reward, digits=2)) (SE)")
            end
            
            # Create DataFrame and save to CSV
            df = DataFrame(
                Policy = [r[1] for r in results_data],
                MeanProfit = [r[2] for r in results_data],
                StdProfit = [r[3] for r in results_data],
                MeanEmissions = [r[4] for r in results_data],
                StdEmissions = [r[5] for r in results_data],
                MeanDiscountedReward = [r[6] for r in results_data],
                StdDiscountedReward = [r[7] for r in results_data],
                StdErrorDiscountedReward = [r[8] for r in results_data]
            )
            
            # Save CSV file
            csv_filename = joinpath(output_dir, "planning_$(planning_model)_evaluation_$(eval_model).csv")
            CSV.write(csv_filename, df)
            
            println("  📊 Results saved to: $csv_filename")
        end
    end
    
    println("\n✅ Price model comparison complete!")
    println("📊 Results saved to: $output_dir")
    println("📁 Generated $(total_configs) CSV files for all price model combinations")
    
    return output_dir
end

# Run the comparison
output_directory = run_price_model_comparison()

println("="^80)
println("🎯 PRICE MODEL COMPARISON COMPLETE")
println("="^80)
println("Generated 16 CSV files (4 planning models × 4 evaluation models)")
println("Each file contains results for all 6 policies with mean and standard deviation")
println("Files saved in: $output_directory")
println("="^80)
