using Statistics
using Printf
using JLD2
using Flux
using ColorSchemes
using CairoMakie
using OceanTurb
using FreeConvection

using OrdinaryDiffEq: ROCK4
using Flux.Losses: mse
using Oceananigans.Units: hours, days

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

Δt₀ = 600  # Output frequency of LES.
les_solutions = Dict(id => ds["T"][1, 1, 1:Nz, :] |> Array for (id, ds) in data.coarse_training_datasets)

function true_T_solution(ds)
    _, _, Nz, _ = size(ds["T"])
    T_LES = ds["T"][1, 1, 1:Nz, :] |> Array
    return T_LES
end

function convective_adjustment_timeseries_loss(datasets, K_CA, Δt_scaling_factor; output_dir=".")
    Δt = Δt₀ * Δt_scaling_factor
    ca_solutions = Dict(id => oceananigans_convective_adjustment(ds, Δt=Δt, K=K_CA; output_dir) for (id, ds) in datasets)

    loss = mean(mse(ca_solutions[id].T, true_T_solution(ds)[:, 1:Δt_scaling_factor:end]) for (id, ds) in datasets)

    @info "K_CA=$K_CA, Δt_scaling_factor=$Δt_scaling_factor, Δt=$Δt, loss=$loss"

    return loss
end

function convective_adjustment_final_loss(datasets, K_CA, Δt; output_dir=".")
    ca_solutions = Dict(id => oceananigans_convective_adjustment(ds, Δt=Δt, K=K_CA; output_dir) for (id, ds) in datasets)

    loss = mean(mse(ca_solutions[id].T[:, end], true_T_solution(ds)[:, end]) for (id, ds) in datasets)

    @info "K_CA=$K_CA, Δt_scaling_factor=$(Δt/Δt₀), Δt=$Δt, loss=$loss"

    return loss
end

function nde_timeseries_loss(datasets, K_CA, nn_filepath, Δt_scaling_factor; output_dir=".")
    Δt = Δt₀ * Δt_scaling_factor
    nde_solutions = Dict(id => oceananigans_convective_adjustment_with_neural_network(ds, Δt=Δt, K=K_CA; nn_filepath, output_dir) for (id, ds) in datasets)

    loss = mean(mse(nde_solutions[id].T, true_T_solution(ds)[:, 1:Δt_scaling_factor:end]) for (id, ds) in datasets)

    @info "K_CA=$K_CA, Δt_scaling_factor=$(Δt/Δt₀), Δt=$Δt, loss=$loss"

    return loss
end

function nde_final_loss(datasets, K_CA, nn_filepath, Δt; output_dir=".")
    nde_solutions = Dict(id => oceananigans_convective_adjustment_with_neural_network(ds, Δt=Δt, K=K_CA; nn_filepath, output_dir) for (id, ds) in datasets)

    loss = mean(mse(nde_solutions[id].T[:, end], true_T_solution(ds)[:, end]) for (id, ds) in datasets)

    @info "K_CA=$K_CA, Δt_scaling_factor=$(Δt/Δt₀), Δt=$Δt, loss=$loss"

    return loss
end

function kpp_timeseries_loss(datasets, Δt_scaling_factor)
    Δt = Δt₀ * Δt_scaling_factor
    times = range(0, 8days, step=Δt)

    kpp_parameters = OceanTurb.KPP.Parameters(CSL=2/3, CNL=5.0, Cb_T=0.16, CKE=8.0)
    kpp_solutions = Dict(id => free_convection_kpp(ds; parameters=kpp_parameters, Δt=Δt, times=times) for (id, ds) in datasets)

    loss = mean(mse(kpp_solutions[id].T, les_solutions[id][:, 1:Δt_scaling_factor:end]) for id in 1:9)

    @info "Δt_scaling_factor=$Δt_scaling_factor, Δt=$Δt, loss=$loss"

    return loss
end

function kpp_final_loss(datasets, Δt)
    times = range(0, 8days, step=Δt)

    kpp_parameters = OceanTurb.KPP.Parameters(CSL=2/3, CNL=5.0, Cb_T=0.16, CKE=8.0)
    kpp_solutions = Dict(id => free_convection_kpp(ds; parameters=kpp_parameters, Δt=Δt, times=times) for (id, ds) in datasets)

    loss = mean(mse(kpp_solutions[id].T[:, end], les_solutions[id][:, end]) for id in 1:9)

    @info "Δt_scaling_factor=$(Δt/Δt₀), Δt=$Δt, loss=$loss"

    return loss
end

Nt = length(datasets[1]["T"].times) - 1
Δt_scaling_factors = [n for n in 1:100 if isinteger(Nt/n)]
Δts = [s*Δt₀ for s in vcat(range(1, 10, step=1), range(10, 100, step=10))]

kpp_losses_ts = [kpp_timeseries_loss_on_datasets(data.coarse_training_datasets, Δ) for Δ in Δt_scaling_factors]
kpp_losses_final = [kpp_final_loss_on_datasets(data.coarse_training_datasets, Δt) for Δt in Δts]

K_CA = 100  # Try increasing even more for NDE?
ca_losses_ts = [convective_adjustment_timeseries_loss(data.coarse_training_datasets, K_CA, Δ) for Δ in Δt_scaling_factors]
ca_losses_final = [convective_adjustment_final_loss(data.coarse_training_datasets, K_CA, Δt) for Δt in Δts]

nn_filepath = joinpath("trained_on_timeseries_dense_default", "neural_network_trained_on_timeseries.jld2")
nde_losses_ts = [nde_timeseries_loss(data.coarse_training_datasets, K_CA, nn_filepath, Δ) for Δ in Δt_scaling_factors]
nde_losses_final = [nde_final_loss(data.coarse_training_datasets, K_CA, nn_filepath, Δt) for Δt in Δts]

begin
    colors = ColorSchemes.julia.colors

    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, xlabel="Time step Δt (hours)", ylabel="Loss ℒ₂")

    scatter!(ax, Δt_scaling_factors .* Δt₀ ./ hours, kpp_losses_ts, color=colors[1])
    lines!(ax, Δts ./ hours, kpp_losses_final, color=colors[1])

    scatter!(ax, Δt_scaling_factors .* Δt₀ ./ hours, ca_losses_ts, color=colors[2])
    lines!(ax, Δts ./ hours, ca_losses_final, color=colors[2])

    scatter!(ax, Δt_scaling_factors .* Δt₀ ./ hours, nde_losses_ts, color=colors[3])
    lines!(ax, Δts ./ hours, nde_losses_final, color=colors[3])

    fig
end
