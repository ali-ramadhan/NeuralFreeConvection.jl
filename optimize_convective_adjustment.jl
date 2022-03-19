using Statistics
using Printf
using JLD2
using Flux
using BenchmarkTools
using CairoMakie
using FreeConvection

using OrdinaryDiffEq: ROCK4
using Flux.Losses: mse

output_dir = "free_convection_nde_dense_default"
nn_filepath = joinpath(output_dir, "free_convection_trained_neural_network.jld2")

file = jldopen(nn_filepath, "r")
T_scaling = file["T_scaling"]
wT_scaling = file["wT_scaling"]
close(file)

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

function run_convective_adjustment_on_datasets(datasets, NN, T_scaling, wT_scaling, K_CA)
    nde_params = Dict(id => ConvectiveAdjustmentNDEParameters(ds, T_scaling, wT_scaling, K_CA) for (id, ds) in datasets)

    ca_solutions = Dict(id => solve_nde(ds, NN, ConvectiveAdjustmentNDE, nde_params[id], ROCK4(), T_scaling, wT_scaling) for (id, ds) in datasets)
    b = @benchmark Dict(id => solve_nde(ds, $NN, ConvectiveAdjustmentNDE, $nde_params[id], ROCK4(), $T_scaling, $wT_scaling) for (id, ds) in $datasets)
    runtime = median(b).time / 1e9

    function true_T_solution(ds)
        _, _, Nz, _ = size(ds["T"])
        T_LES = ds["T"][1, 1, 1:Nz, :] |> Array
        return T_LES
    end

    loss = mean(mse(ca_solutions[id].T, true_T_solution(ds)) for (id, ds) in datasets)

    @info "K_CA=$K_CA, loss=$loss, runtime=$runtime"

    return loss, runtime
end

# Use a neural network with zero weights so it doesn't affect the solution, but we get a accurate estimates of the NDE's runtime.

NN = Chain(Dense(Nz, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))

for p in params(NN)
    p .= 0
end

N = 100
K_CAs = 10 .^ range(-3, 1, length=N)

losses = zeros(N)
runtimes = zeros(N)

for (i, K_CA) in enumerate(K_CAs)
    @info "Running convective adjustment with K_CA=$K_CA..."
    loss, runtime = run_convective_adjustment_on_datasets(data.coarse_training_datasets, NN, T_scaling, wT_scaling, K_CA)
    losses[i] = loss
    runtimes[i] = runtime
end

begin
    fig = Figure()

    best_K_CA = K_CAs[argmin(losses .* runtimes)]
    @show best_K_CA

    ax1 = Axis(fig[1, 1], ylabel="MSE loss", xscale=log10, yscale=log10)
    lines!(ax1, K_CAs, losses)
    vlines!(ax1, best_K_CA, color=(:red, 0.5))

    ax2 = Axis(fig[2, 1], xlabel="K_CA", ylabel="Runtime (seconds)", xscale=log10)
    lines!(ax2, K_CAs, runtimes)
    vlines!(ax2, best_K_CA, color=(:red, 0.5))

    xlims!(ax1, (K_CAs[1], K_CAs[end]))
    xlims!(ax2, (K_CAs[1], K_CAs[end]))

    save("figureC_optimal_convective_adjustment_parameter.png", fig, px_per_unit=2)
end
