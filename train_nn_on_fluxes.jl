using Logging
using Printf
using Random
using Statistics
using LinearAlgebra

using ArgParse
using LoggingExtras
using Flux
using JLD2
using CairoMakie

using Oceananigans
using OceanParameterizations
using FreeConvection

using OrdinaryDiffEq: ROCK4
using FreeConvection: inscribe_history

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
ENV["GKSwstype"] = "100"

LinearAlgebra.BLAS.set_num_threads(1)

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--grid-points"
            help = "Number of vertical grid points in the trained neural differential equation (LES data will be coarse-grained to this resolution)."
            default = 32
            arg_type = Int

        "--base-parameterization"
            help = "Base parameterization to use for the NDE. Options: nothing, convective_adjustment"
            default = "nothing"
            arg_type = String

        "--output-directory"
            help = "Output directory filepath."
            default = joinpath(@__DIR__, "testing")
            arg_type = String

        "--training-simulations"
            help = "Simulation IDs (list of integers separated by spaces) to train the neural differential equation on." *
                   "All other simulations will be used for testing/validation."
            action = :append_arg
            nargs = '+'
            arg_type = Int
            range_tester = (id -> id in FreeConvection.SIMULATION_IDS)

        "--epochs"
            help = "Number of epochs per optimizer to train on the full time series."
            default = 10
            arg_type = Int

        "--animate-training-data"
            help = "Produce gif and mp4 animations of each training simulation's data."
            action = :store_true
    end

    return parse_args(settings)
end


@info "Parsing command line arguments..."

args = parse_command_line_arguments()

base_param = args["base-parameterization"]
use_convective_adjustment = base_param == "convective_adjustment"
use_missing_fluxes = use_convective_adjustment

Nz = args["grid-points"]
epochs = args["epochs"]

ids_train = args["training-simulations"][1]
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)
validate_simulation_ids(ids_train, ids_test)

output_dir = args["output-directory"]
mkpath(output_dir)

animate_training_simulations = args["animate-training-data"]

# Save command line arguments used to an executable shell script
open(joinpath(output_dir, "train_nn_on_fluxes.sh"), "w") do io
    write(io, "#!/bin/sh\n")
    write(io, "julia " * basename(@__FILE__) * " " * join(ARGS, " ") * "\n")
end

@info "Planting loggers..."

log_filepath = joinpath(output_dir, "training_on_fluxes.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger


@info "Architecting neural network..."

NN = Chain(Dense(Nz, 4Nz, relu),
           Dense(4Nz, 4Nz, relu),
           Dense(4Nz, Nz-1))

for p in params(NN)
    p .*= 1e-5
end

function free_convection_neural_network(input)
    wT_interior = NN(input.temperature)
    wT = cat(input.bottom_flux, wT_interior, input.top_flux, dims=1)
    return wT
end


@info "Loading training data..."

data = load_data(ids_train, ids_test, Nz)

training_datasets = data.training_datasets
coarse_training_datasets = data.coarse_training_datasets


if use_convective_adjustment
    @info "Computing convective adjustment solutions and fluxes (and missing fluxes)..."

    for (id, ds) in data.coarse_datasets
        sol = oceananigans_convective_adjustment(ds; output_dir)

        grid = ds["T"].grid
        times = ds["T"].times

        ds.fields["T_param"] = FieldTimeSeries(grid, (Center, Center, Center), times, ArrayType=Array{Float32})
        ds.fields["wT_param"] = FieldTimeSeries(grid, (Center, Center, Face), times, ArrayType=Array{Float32})
        ds.fields["wT_missing"] = FieldTimeSeries(grid, (Center, Center, Face), times, ArrayType=Array{Float32})

        ds.fields["T_param"][1, 1, :, :] .= sol.T
        ds.fields["wT_param"][1, 1, :, :] .= sol.wT

        ds.fields["wT_missing"].data .= ds.fields["wT"].data .- ds.fields["κₑ_∂z_T"].data .- ds.fields["wT_param"].data

    end
end


if animate_training_simulations
    @info "Animating ⟨T⟩(z,t) and ⟨w'T⟩(z,t) training data..."

    for id in keys(training_datasets)
        filepath = joinpath(output_dir, "free_convection_training_data_$id")
        if !isfile(filepath * ".mp4") || !isfile(filepath * ".gif")
            animate_training_data(training_datasets[id], coarse_training_datasets[id]; filepath, frameskip=5)
        end
    end
end


@info "Wrangling (T, wT) training data..."

input_training_data = wrangle_input_training_data(coarse_training_datasets, use_missing_fluxes=use_missing_fluxes, time_range_skip=1)
output_training_data = wrangle_output_training_data(coarse_training_datasets; use_missing_fluxes, time_range_skip=1)


@info "Scaling features..."

T_training_data = reduce(hcat, input.temperature for input in input_training_data)
wT_training_data = output_training_data

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

input_training_data = [rescale(i, T_scaling, wT_scaling) for i in input_training_data]
output_training_data = wT_scaling.(output_training_data)

training_data = [(input_training_data[n], output_training_data[:, n]) for n in 1:length(input_training_data)] |> shuffle


@info "Removing outliers from training data..."

nn_loss(input, output) = Flux.Losses.mse(free_convection_neural_network(input), output)

# We will remove training dat with loss > max_loss since they pollute the mean.
max_loss = 1

losses₀ = [nn_loss(input, output) for (input, output) in training_data]
outliers = losses₀ .> max_loss
n_outliers = count(outliers)
training_data = training_data[.!outliers]

@info @sprintf("Removed %d/%d (%.2f%%) of the training data.", n_outliers, length(losses₀), 100 * n_outliers/length(losses₀))


@info "Batching training data..."

n_training_data = length(training_data)
data_loader = Flux.Data.DataLoader(training_data, batchsize=n_training_data, shuffle=true)

n_obs = data_loader.nobs
batch_size = data_loader.batchsize
n_batches = ceil(Int, n_obs / batch_size)
@info "Training data loader contains $n_obs pairs of observations (batch size = $batch_size)."


@info "Training neural network on fluxes: ⟨T⟩(z) -> ⟨w′T′⟩(z) mapping..."

nn_training_set_loss(training_data) = mean(nn_loss(input, output) for (input, output) in training_data)

function nn_callback()
    losses = [nn_loss(input, output) for (input, output) in training_data]

    mean_loss = mean(losses)
    median_loss = median(losses)

    @info @sprintf("Training free convection neural network... training set MSE loss: mean_loss::%s = %.10e, median_loss = %.10e",
                   typeof(mean_loss), mean_loss, median_loss)

    return mean_loss, median_loss
end

history_filepath = joinpath(output_dir, "neural_network_history_trained_on_fluxes.jld2")

optimizers = [ADAM()]
for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader)
    @info "Training heat flux neural network with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"

    t₀ = time_ns()
    # Flux.train!(nn_loss, Flux.params(NN), mini_batch, opt)
    Flux.train!(nn_training_set_loss, Flux.params(NN), [training_data], opt)
    runtime = (time_ns() - t₀) * 1e-9

    mean_loss, median_loss = nn_callback()
    inscribe_history(history_filepath, e, neural_network=NN; mean_loss, median_loss, runtime)
end


@info "Saving trained neural network weights to disk..."

nn_filepath = joinpath(output_dir, "neural_network_trained_on_fluxes.jld2")

jldopen(nn_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling
end


@info "Plotting loss history..."

file = jldopen(history_filepath, "r")

mean_losses = [file["mean_loss/$e"] for e in 1:epochs]
median_losses = [file["median_loss/$e"] for e in 1:epochs]

close(file)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Epochs", ylabel="Loss")
    lines!(ax, 1:epochs, mean_losses, label="mean")
    lines!(ax, 1:epochs, median_losses, label="median")
    axislegend(ax, position=:rt)
    filepath = joinpath(output_dir, "loss_history_trained_on_fluxes.png")
    save(filepath, fig, px_per_unit=2)
end


@info "Animating learned fluxes..."

function animate_fluxes(ds, NN, T_scaling, wT_scaling; filepath, title="", framerate=15)
    T = interior(ds["T"])
    wT = interior(ds["wT"])
    wT_les = interior(ds["κₑ_∂z_T"])
    wT_param = interior(ds["wT_param"])
    wT_missing = interior(ds["wT_missing"])

    wT_min = minimum(quantile(x[:], 0.02) for x in (wT, wT_les, wT_param, wT_missing))
    wT_max = maximum(quantile(x[:], 0.98) for x in (wT, wT_les, wT_param, wT_missing))

    Nz = size(wT, 3) - 1
    Nt = size(wT, 4)
    zf = znodes(ds["wT"])

    fig = Figure(resolution=(600, 800))
    ax = Axis(fig[1, 1], title=title, xlabel="heat flux", ylabel="z (m)")

    n = Observable(1)

    wT_n = @lift wT[1, 1, :, $n]
    wT_les_n = @lift -wT_les[1, 1, :, $n]
    wT_total_n = @lift wT[1, 1, :, $n] - wT_les[1, 1, :, $n]
    wT_param_n = @lift wT_param[1, 1, :, $n]
    wT_missing_n = @lift wT_missing[1, 1, :, $n]
    wT_missing_NN_n = @lift T[1, 1, :, $n] |> T_scaling |> NN |> inv(wT_scaling)

    lines!(ax, wT_n, zf, label="wT")
    lines!(ax, wT_les_n, zf, label="LES")
    lines!(ax, wT_total_n, zf, label="wT + LES")
    lines!(ax, wT_param_n, zf, label="param")
    lines!(ax, wT_missing_n, zf, label="missing")
    lines!(ax, wT_missing_NN_n, zf[2:end-1], label="NN")

    xlims!(ax, (wT_min - abs(wT_min), wT_max + 0.1 * abs(wT_max)))
    ylims!(ax, (zf[1], zf[end]))

    axislegend(ax, position=:rb)

    record(fig, filepath, 1:Nt; framerate) do time_index
        @info "Animating $filepath frame $time_index/$Nt..."
        n[] = time_index
    end
end

for (id, ds) in data.coarse_datasets
    filepath = joinpath(output_dir, @sprintf("animating_fluxes_simulation%02d.mp4", id))
    animate_fluxes(ds, NN, T_scaling, wT_scaling; filepath)
end


@info "Computing flux history for each simulation..."

flux_history, flux_loss_history = compute_nn_flux_prediction_history(data.coarse_datasets, nn_filepath, history_filepath)

jldopen(history_filepath, "a") do file
    file["flux_history"] = flux_history
    file["flux_loss_history"] = flux_loss_history
end

function clean_flux_loss_history(lh; max_loss=1)
    lh = copy(lh)
    outliers = lh .> max_loss
    lh[outliers] .= 0
    return lh
end


@info "Plotting flux loss history for each simulation...."

begin
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, title="Flux loss history on training simulations", xlabel="Epochs", ylabel="MSE loss (fluxes)")
    for id in 1:9
        lines!(mean(clean_flux_loss_history(flux_loss_history[id]), dims=2)[:], label="$id")
    end
    xlims!(ax, (0, epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "flux_loss_history_trained_on_fluxes_training.png")
    save(filepath, fig, px_per_unit=2)

    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, title="Flux loss history on validation simulations", xlabel="Epochs", ylabel="MSE loss (fluxes)")
    for id in 10:21
        lines!(mean(clean_flux_loss_history(flux_loss_history[id]), dims=2)[:], label="$id")
    end
    xlims!(ax, (0, epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "flux_loss_history_trained_on_fluxes_validation.png")
    save(filepath, fig, px_per_unit=2)
end


@info "Computing NDE solution and loss history for each simulation..."

K_CA = 2  # Optimal value from optimize_convective_adujstment.jl
nde_params = Dict(id => ConvectiveAdjustmentNDEParameters(ds, T_scaling, wT_scaling, K_CA) for (id, ds) in data.coarse_datasets)
solution_history, solution_loss_history = compute_nde_solution_history(data.coarse_datasets, ConvectiveAdjustmentNDE, nde_params, ROCK4(), nn_filepath, history_filepath)

jldopen(history_filepath, "a") do file
    file["solution_history"] = solution_history
    file["solution_loss_history"] = solution_loss_history
end

@info "Plotting solution loss history for each simulation...."

begin
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, title="Solution loss history on training simulations", xlabel="Epochs", ylabel="MSE loss (time series)")
    for id in 1:9
        lines!(mean(solution_loss_history[id], dims=2)[:], label="$id")
    end
    xlims!(ax, (0, epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "solution_loss_history_trained_on_fluxes_training.png")
    save(filepath, fig, px_per_unit=2)

    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, title="Solution loss history on validation simulations", xlabel="Epochs", ylabel="MSE loss (time series)")
    for id in 10:21
        lines!(mean(solution_loss_history[id], dims=2)[:], label="$id")
    end
    xlims!(ax, (0, epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "solution_loss_history_trained_on_fluxes_validation.png")
    save(filepath, fig, px_per_unit=2)
end
