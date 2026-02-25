#!/usr/bin/env julia

"""
Plot Average Belief Uncertainty over Time for Different Policies

This script analyzes belief uncertainty evolution across time steps for:
- POMCPOW
- OneMineProfitMaximization  
- ExploreOnly

It creates a plot with error bars showing mean ± std deviation.
"""

using LiPOMDPs
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner
using PGFPlots

# Import specific functions from pomdp.jl
import LiPOMDPs: calculate_site_production, calculate_total_production, compute_costs, compute_emissions

# =============================================================================
# POLICY CREATION FUNCTION
# =============================================================================

function create_policy_for_uncertainty(policy_type::String, pomdp::LiPOMDP)
    """Create a policy of the specified type for uncertainty analysis."""
    if policy_type == "POMCPOW"
        solver = POMCPOWSolver(
            tree_queries = 2500,
            max_depth = 25,
            criterion = MaxUCB(15.0),
            enable_action_pw = true,
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 3.0,
            alpha_observation = 0.3,
            estimate_value = FORollout(DynamicProfitMaximizerPolicy(pomdp))
        )
        return solve(solver, pomdp)
    elseif policy_type == "OneMineProfitMaximization"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp)
    else
        error("Unknown policy type: $policy_type")
    end
end

# =============================================================================
# BELIEF UNCERTAINTY CALCULATION
# =============================================================================

function calculate_belief_uncertainty(belief::LiBelief)
    """
    Calculate average belief uncertainty across all deposits.
    Uncertainty is measured as the standard deviation of the deposit distributions.
    """
    uncertainties = [std(dist) for dist in belief.deposits_distribution]
    return mean(uncertainties)
end

# =============================================================================
# RUN SIMULATION AND COLLECT BELIEF UNCERTAINTY
# =============================================================================

function collect_belief_uncertainty_over_time(
    pomdp::LiPOMDP, 
    policy, 
    updater::LiBeliefUpdater,
    n_sims::Int=10;
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    """
    Run simulations and collect belief uncertainty at each time step.
    Returns a matrix where rows are time steps and columns are simulations.
    """
    
    max_time = pomdp.time_horizon
    uncertainty_matrix = zeros(max_time, n_sims)
    
    for sim in 1:n_sims
        Random.seed!(sim + 5000)  # Different seeds for each simulation
        
        # Initialize
        s0, b0 = create_initial_belief(pomdp)
        b = b0
        s = s0
        
        for t in 1:max_time
            # Calculate and store belief uncertainty at this time step
            uncertainty_matrix[t, sim] = calculate_belief_uncertainty(b)
            
            # Choose action
            if policy isa POMCPOWPlanner
                a = action(policy, b)
            else
                a = action(policy, b)
            end
            
            # Generate next state and observation
            sp, o, r = gen(pomdp, s, a, rng)
            
            # Check for terminal state
            if isterminal(pomdp, sp)
                # Fill remaining time steps with the last uncertainty value
                for remaining_t in (t+1):max_time
                    uncertainty_matrix[remaining_t, sim] = uncertainty_matrix[t, sim]
                end
                break
            end
            
            # Update belief
            b = update(updater, b, a, o)
            s = sp
        end
    end
    
    return uncertainty_matrix
end

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

function analyze_belief_uncertainty(
    policies::Vector{String},
    alpha::Float64=0.5;
    n_sims::Int=10,
    time_horizon::Int=29
)
    """
    Analyze belief uncertainty over time for multiple policies.
    Returns a dictionary with statistics for each policy.
    """
    
    println("🔍 BELIEF UNCERTAINTY ANALYSIS")
    println("="^60)
    println("Alpha: $alpha")
    println("Simulations per policy: $n_sims")
    println("Time horizon: $time_horizon")
    println()
    
    results = Dict{String, Dict}()
    
    for policy_type in policies
        println("Analyzing policy: $policy_type")
        
        # Create POMDP
        pomdp = LiPOMDP(price_model_type=2, time_horizon=time_horizon, alpha=alpha)
        
        # Create belief updater
        updater = LiBeliefUpdater(P=pomdp)
        
        # Create policy
        policy = create_policy_for_uncertainty(policy_type, pomdp)
        
        # Collect uncertainty data
        uncertainty_matrix = collect_belief_uncertainty_over_time(pomdp, policy, updater, n_sims)
        
        # Calculate statistics
        mean_uncertainty = vec(mean(uncertainty_matrix, dims=2))
        std_uncertainty = vec(std(uncertainty_matrix, dims=2))
        
        results[policy_type] = Dict(
            "mean" => mean_uncertainty,
            "std" => std_uncertainty,
            "raw" => uncertainty_matrix
        )
        
        println("  ✓ Mean final uncertainty: $(round(mean_uncertainty[end], digits=2))")
        println("  ✓ Std final uncertainty: $(round(std_uncertainty[end], digits=2))")
        println()
    end
    
    return results
end

# =============================================================================
# HELPER FUNCTION FOR STANDALONE LATEX DOCUMENT
# =============================================================================

function create_standalone_tex(tikz_code::String, tex_file::String)
    """Create a standalone LaTeX document that can be compiled to PDF."""
    standalone_doc = """
\\documentclass[tikz]{standalone}
\\usepackage{pgfplots}
\\usepackage{amsmath}
\\pgfplotsset{compat=1.16}

\\begin{document}
$tikz_code
\\end{document}
"""
    open(tex_file, "w") do f
        write(f, standalone_doc)
    end
end

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

function plot_belief_uncertainty(results::Dict, output_file::String="belief_uncertainty.pdf")
    """
    Create a PGFPlots visualization of belief uncertainty over time with error bars.
    """
    
    println("📊 Creating plot...")
    
    # Define colors and styles for each policy
    policy_styles = Dict(
        "POMCPOW" => ("blue", "solid", "*", "POMCPOW"),
        "OneMineProfitMaximization" => ("red", "dashed", "square*", "One Mine Profit Max"),
        "ExploreOnly" => ("green!70!black", "dotted", "triangle*", "Explore Only")
    )
    
    # Create axis first
    ax = Axis(
        xlabel = "Time Step",
        ylabel = "Avg. Deposit Belief Uncertainty (tonnes Li)",
        title = "Belief Uncertainty Evolution by Policy",
        legendPos = "north east",
        width = "14cm",
        height = "9cm",
        style = "grid=major, grid style={line width=.2pt, draw=gray!20}, legend style={font=\\small}"
    )
    
    # Add plots to axis
    for (policy_name, data) in results
        mean_vals = data["mean"]
        std_vals = data["std"]
        time_steps = collect(1:length(mean_vals))
        
        color, line_style, marker, display_name = policy_styles[policy_name]
        
        # Create error bars plot using PGFPlots.ErrorBars
        plot = Plots.Linear(
            time_steps, 
            mean_vals,
            errorBars = ErrorBars(; y = std_vals),
            style = "$(color), thick, $(line_style), mark=$(marker), mark size=2.5pt, error bars/.cd, y dir=both, y explicit",
            legendentry = display_name
        )
        push!(ax, plot)
    end
    
    # Get the TikZ code from the axis
    tikz_code = string(ax)
    
    # Create standalone TEX file (with full LaTeX document structure)
    tex_file = replace(output_file, ".pdf" => ".tex")
    create_standalone_tex(tikz_code, tex_file)
    println("✅ Standalone TEX file saved to: $tex_file")
    
    # Try to compile to PDF
    println("📝 Compiling LaTeX to PDF...")
    try
        cd(dirname(abspath(output_file))) do
            # Run pdflatex
            run(`pdflatex -interaction=nonstopmode $(basename(tex_file))`)
            
            pdf_name = basename(output_file)
            if isfile(pdf_name)
                println("✅ PDF successfully compiled: $output_file")
                
                # Clean up auxiliary files
                for ext in [".aux", ".log", ".out"]
                    aux_file = replace(pdf_name, ".pdf" => ext)
                    isfile(aux_file) && rm(aux_file)
                end
            else
                println("⚠️  PDF not found after compilation")
                println("   You can manually compile with: cd $(dirname(abspath(output_file))) && pdflatex $(basename(tex_file))")
            end
        end
    catch compile_error
        println("⚠️  LaTeX compilation failed: $compile_error")
        println("   TEX file is available at: $tex_file")
        println("   You can manually compile with: cd $(dirname(abspath(output_file))) && pdflatex $(basename(tex_file))")
    end
    
    return ax
end

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

println("🚀 Starting Belief Uncertainty Analysis")
println("="^60)

# Configuration
policies_to_analyze = ["POMCPOW", "OneMineProfitMaximization", "ExploreOnly"]
alpha_value = 0.5
n_simulations = 30  # Number of simulations to average over
time_horizon = 29

println("Policies: $policies_to_analyze")
println("Running analysis...")
println()

# Run the analysis
results = analyze_belief_uncertainty(
    policies_to_analyze,
    alpha_value,
    n_sims=n_simulations,
    time_horizon=time_horizon
)

# Create the plot
output_plot = "outputs/belief_uncertainty_comparison.pdf"
plot_belief_uncertainty(results, output_plot)

println("\n" * "="^60)
println("✅ ANALYSIS COMPLETE!")
println("📈 Results:")
for (policy, data) in results
    println("  • $policy:")
    println("    - Initial uncertainty: $(round(data["mean"][1], digits=2)) ± $(round(data["std"][1], digits=2))")
    println("    - Final uncertainty: $(round(data["mean"][end], digits=2)) ± $(round(data["std"][end], digits=2))")
    println("    - Reduction: $(round(data["mean"][1] - data["mean"][end], digits=2)) tonnes Li")
end
println("="^60)

