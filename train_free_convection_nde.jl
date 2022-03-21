using Logging
using Printf
using Random
using Statistics
using LinearAlgebra

using ArgParse
using LoggingExtras
using DataDeps
using Flux
using JLD2
using OrdinaryDiffEq
using Zygote
using CairoMakie

using Oceananigans
using OceanParameterizations
using FreeConvection

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

        "--time-stepper"
            help = "DifferentialEquations.jl time stepping algorithm to use."
            default = "Tsit5"
            arg_type = String

        "--training-simulations"
            help = "Simulation IDs (list of integers separated by spaces) to train the neural differential equation on." *
                   "All other simulations will be used for testing/validation."
            action = :append_arg
            nargs = '+'
            arg_type = Int
            range_tester = (id -> id in FreeConvection.SIMULATION_IDS)

        "--burn-in-epochs"
            help = "Number of epocepochshs to train on the partial time series."
            default = 0
            arg_type = Int

        "--training-epochs"
            help = "Number of epochs per optimizer to train on the full time series."
            default = 10
            arg_type = Int

        "--neural-network-architecture"
            help = "Chooses from a set of pre-defined neural network architectures. Options: dense-default, dense-deeper, dense-wider, conv-2, conv-4"
            default = "dense-default"
            arg_type = String

        "--animate-training-data"
            help = "Produce gif and mp4 animations of each training simulation's data."
            action = :store_true

        "--output-directory"
            help = "Output directory filepath."
            default = joinpath(@__DIR__, "testing")
            arg_type = String
    end

    return parse_args(settings)
end


@info "Parsing command line arguments..."

args = parse_command_line_arguments()

nde_type = Dict(
    "nothing" => FreeConvectionNDE,
    "convective_adjustment" => ConvectiveAdjustmentNDE
)

Nz = args["grid-points"]
NDEType = nde_type[args["base-parameterization"]]
algorithm = Meta.parse(args["time-stepper"] * "()") |> eval

nn_architecture = args["neural-network-architecture"]
burn_in_epochs = args["burn-in-epochs"]
full_epochs = args["training-epochs"]

ids_train = args["training-simulations"][1]
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)
validate_simulation_ids(ids_train, ids_test)

animate_training_simulations = args["animate-training-data"]

output_dir = args["output-directory"]
mkpath(output_dir)

# Save command line arguments used to an executable shell script
open(joinpath(output_dir, "train_free_convection_nde.sh"), "w") do io
    write(io, "#!/bin/sh\n")
    write(io, "julia " * basename(@__FILE__) * " " * join(ARGS, " ") * "\n")
end


@info "Planting loggers..."

log_filepath = joinpath(output_dir, "training.log")
TeeLogger(
    OceananigansLogger(),
    MinLevelLogger(FileLogger(log_filepath), Logging.Info)
) |> global_logger


@info "Architecting neural network..."

if nn_architecture == "dense-default"
    NN = Chain(Dense(Nz, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, Nz-1))
elseif nn_architecture == "dense-wider"
    NN = Chain(Dense(Nz, 8Nz, relu),
               Dense(8Nz, 8Nz, relu),
               Dense(8Nz, Nz-1))
elseif nn_architecture == "dense-deeper"
    NN = Chain(Dense(Nz, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, Nz-1))
elseif nn_architecture == "conv-2"
    conv = 2
    NN = Chain(x -> reshape(x, Nz, 1, 1, 1),
               Conv((conv, 1), 1 => 1, relu),
               x -> reshape(x, Nz-conv+1),
               Dense(Nz-conv+1, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, Nz-1))
elseif nn_architecture == "conv-4"
    conv = 4
    NN = Chain(x -> reshape(x, Nz, 1, 1, 1),
               Conv((conv, 1), 1 => 1, relu),
               x -> reshape(x, Nz-conv+1),
               Dense(Nz-conv+1, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, Nz-1))
else
    @error "Invalid neural network architecture: $nn_architecture"
end

for p in params(NN)
    p .*= 1e-5
end


@info "Loading training data..."

data = load_data(ids_train, ids_test, Nz)

training_datasets = data.training_datasets
coarse_training_datasets = data.coarse_training_datasets
coarse_datasets = data.coarse_datasets


@info "Computing convective adjustment solutions and fluxes (and missing fluxes)..."

for (id, ds) in coarse_datasets
    sol = oceananigans_convective_adjustment(ds; output_dir)

    grid = ds["T"].grid
    times = ds["T"].times

    ds.fields["T_param"] = FieldTimeSeries(grid, (Center, Center, Center), times, ArrayType=Array{Float32})
    ds.fields["wT_param"] = FieldTimeSeries(grid, (Center, Center, Face), times, ArrayType=Array{Float32})
    ds.fields["wT_missing"] = FieldTimeSeries(grid, (Center, Center, Face), times, ArrayType=Array{Float32})

    ds.fields["T_param"][1, 1, :, :] .= sol.T
    ds.fields["wT_param"][1, 1, :, :] .= sol.wT
    ds.fields["wT_missing"].data[1, 1, :, :] .= ds.fields["wT"].data[1, 1, :, :] .- ds.fields["wT_param"].data[1, 1, :, :]
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

input_training_data = wrangle_input_training_data(coarse_training_datasets, use_missing_fluxes=false)
output_training_data = wrangle_output_training_data(coarse_training_datasets, use_missing_fluxes=false)


@info "Scaling features..."

T_training_data = reduce(hcat, input.temperature for input in input_training_data)
wT_training_data = output_training_data

@assert size(wT_training_data, 1) == size(T_training_data, 1) + 1
@assert size(wT_training_data, 2) == size(T_training_data, 2)

T_scaling = ZeroMeanUnitVarianceScaling(T_training_data)
wT_scaling = ZeroMeanUnitVarianceScaling(wT_training_data)

input_training_data = [rescale(i, T_scaling, wT_scaling) for i in input_training_data]
output_training_data = wT_scaling.(output_training_data)


@info "Training neural differential equation on incrementally increasing time spans..."

history_filepath = joinpath(output_dir, "neural_network_history_trained_on_timeseries.jld2")

if burn_in_epochs > 0
    training_iterations = (1:20, 1:5:101, 1:10:201, 1:20:401, 1:40:801)
    opt = ADAM()

    for iterations in training_iterations
        @info "Training free convection NDE with iterations=$iterations for $burn_in_epochs epochs  with $(typeof(opt))(η=$(opt.eta))..."
        train_neural_differential_equation!(NN, NDEType, algorithm, coarse_training_datasets, T_scaling, wT_scaling,
                                            iterations, opt, burn_in_epochs, history_filepath=history_filepath)
    end
end


@info "Training the neural differential equation on the entire solution..."

K_CA = 2  # Optimal value from optimize_convective_adujstment.jl
nde_params = Dict(id => ConvectiveAdjustmentNDEParameters(ds, T_scaling, wT_scaling, K_CA) for (id, ds) in data.coarse_datasets)

training_iterations = 1:9:1153
optimizers = [ADAM()]

for opt in optimizers
    @info "Training free convection NDE with iterations=$training_iterations for $full_epochs epochs with $(typeof(opt))(η=$(opt.eta))..."

    global NN
    t₀ = time_ns()

    NN = train_neural_differential_equation!(NN, NDEType, nde_params, algorithm, coarse_training_datasets, T_scaling,
                                             training_iterations, opt, full_epochs; history_filepath)

    runtime = (time_ns() - t₀) * 1e-9
    inscribe_history(history_filepath, 1; runtime)
end


@info "Saving trained neural network weights to disk..."

nn_filepath = joinpath(output_dir, "neural_network_trained_on_timeseries.jld2")

jldopen(nn_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling
end


@info "Plotting loss history..."

file = jldopen(history_filepath, "r")

mean_losses = [file["mean_loss/$e"] for e in 1:full_epochs]

close(file)

begin
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, xlabel="Epochs", ylabel="Loss")
    lines!(ax, 1:full_epochs, mean_losses)
    xlims!(ax, (0, full_epochs))
    filepath = joinpath(output_dir, "loss_history_trained_on_timeseries.png")
    save(filepath, fig, px_per_unit=2)
end


@info "Computing flux history for each simulation..."

flux_history, flux_loss_history = compute_nn_flux_prediction_history(data.coarse_datasets, nn_filepath, history_filepath)

jldopen(history_filepath, "a") do file
    file["flux_history"] = flux_history
    file["flux_loss_history"] = flux_loss_history
end


@info "Plotting flux loss history for each simulation...."

begin
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, title="Flux loss history on training simulations", xlabel="Epochs", ylabel="MSE loss (fluxes)")
    for id in 1:9
        lines!(mean(flux_loss_history[id], dims=2)[:], label="$id")
    end
    xlims!(ax, (0, full_epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "flux_loss_history_trained_on_timeseries_training.png")
    save(filepath, fig, px_per_unit=2)

    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, title="Flux loss history on validation simulations", xlabel="Epochs", ylabel="MSE loss (fluxes)")
    for id in 10:21
        lines!(mean(flux_loss_history[id], dims=2)[:], label="$id")
    end
    xlims!(ax, (0, full_epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "flux_loss_history_trained_on_timeseries_validation.png")
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
    xlims!(ax, (0, full_epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "solution_loss_history_trained_on_timeseries_training.png")
    save(filepath, fig, px_per_unit=2)

    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10, title="Solution loss history on validation simulations", xlabel="Epochs", ylabel="MSE loss (time series)")
    for id in 10:21
        lines!(mean(solution_loss_history[id], dims=2)[:], label="$id")
    end
    xlims!(ax, (0, full_epochs))
    Legend(fig[1, 2], ax, "simulation", framevisible=false)

    filepath = joinpath(output_dir, "solution_loss_history_trained_on_timeseries_validation.png")
    save(filepath, fig, px_per_unit=2)
end


@info "Gathering and computing solutions..."

true_solutions = Dict(id => (T=interior(ds["T"])[1, 1, :, :], wT=interior(ds["wT"])[1, 1, :, :]) for (id, ds) in coarse_datasets)
nde_solutions = Dict(id => solve_nde(ds, NN, NDEType, nde_params[id], algorithm, T_scaling, wT_scaling) for (id, ds) in coarse_datasets)
kpp_solutions = Dict(id => free_convection_kpp(ds) for (id, ds) in coarse_datasets)
tke_solutions = Dict(id => free_convection_tke_mass_flux(ds) for (id, ds) in coarse_datasets)

convective_adjustment_solutions = Dict(id => oceananigans_convective_adjustment(ds, K=K_CA; output_dir) for (id, ds) in coarse_datasets)
oceananigans_solutions = Dict(id => oceananigans_convective_adjustment_with_neural_network(ds, K=K_CA, output_dir=output_dir, nn_filepath=nn_filepath) for (id, ds) in coarse_datasets)


@info "Saving solutions to JLD2..."

solutions_filepath = joinpath(output_dir, "solutions_and_history.jld2")

jldopen(solutions_filepath, "w") do file
    file["grid_points"] = Nz
    file["neural_network"] = NN
    file["T_scaling"] = T_scaling
    file["wT_scaling"] = wT_scaling

    file["true"] = true_solutions
    file["nde"] = nde_solutions
    file["kpp"] = kpp_solutions
    file["tke"] = tke_solutions
    file["convective_adjustment"] = convective_adjustment_solutions
    file["oceananigans"] = oceananigans_solutions

    file["nde_history"] = nde_solution_history
end
