using Printf
using Random
using Statistics
using LinearAlgebra
using Logging

using NCDatasets
using Plots
using Flux
using BSON
using DifferentialEquations
using DiffEqFlux
using Optim

using ClimateSurrogates
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Utils

using Flux.Data: DataLoader

# Should not have saved constant units as strings...
nc_constant(attr) = parse(Float64, attr |> split |> first)

Nz = 32

Qs = [25, 50, 75, 100]
Qs_train = [25, 75]
Qs_test = [50, 100]

ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs)

T_training_data = cat([training_data(ds[Q]["T"], grid_points=Nz) for Q in Qs_train]..., dims=2)

wT_training_data = []

for Q in Qs_train
    ρ₀ = nc_constant(ds[Q].attrib["Reference density"])
    cₚ = nc_constant(ds[Q].attrib["Specific_heat_capacity"])

    wT_training_data_Q = training_data(ds[Q]["wT"], grid_points=Nz+1)

    # We need to add the imposed surface heat flux.
    top_flux = Q / (ρ₀ * cₚ)
    wT_training_data_Q[end, :] .= top_flux

    push!(wT_training_data, wT_training_data_Q)
end

wT_training_data = cat(wT_training_data..., dims=2)

function free_convection_heat_flux_training_data(ds, Qs; grid_points, skip_first=0)
     T = Dict(Q => ds[Q]["T"]  for Q in Qs)
    wT = Dict(Q => ds[Q]["wT"] for Q in Qs)
    Nz, Nt = size(T[Qs[1]])

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    T_inputs, top_flux_inputs, wT_outputs = Dict(), Dict(), Dict()

    for Q in Qs
        ρ₀ = nc_constant(ds[Q], "Reference density")
        cₚ = nc_constant(ds[Q], "Specific_heat_capacity")

        top_flux = Q / (ρ₀ * cₚ)
        bot_flux = 0.0

        inds = 1+skip_first:Nt
        top_flux_inputs[Q] = [top_flux for _ in inds]
        T_inputs[Q] = cat((coarse_grain(T[Q][:, n], grid_points, Cell) for n in inds)..., dims=2)
        wT_outputs[Q] = cat((coarse_grain(wT[Q][:, n], grid_points+1, Face) for n in inds)..., dims=2)

        for n in 1:Nt-skip_first
            wT_outputs[Q][1, n] = bot_flux
            wT_outputs[Q][grid_points+1, n] = top_flux
        end
    end

    top_flux_inputs = cat((top_flux_inputs[Q]  for Q in Qs)..., dims=2)
    T_inputs = cat((T_inputs[Q] for Q in Qs)..., dims=2)
    wT_outputs = cat((wT_outputs[Q] for Q in Qs)..., dims=2)

    μ_T, σ_T = mean(T_inputs), std(T_inputs)
    μ_wT, σ_wT = mean(wT_outputs), std(wT_outputs)

    standardize_T(x) = (x - μ_T) / σ_T
    standardize⁻¹_T(y) = σ_T * y + μ_T
    standardize_wT(x) = (x - μ_wT) / σ_wT
    standardize⁻¹_wT(y) = σ_wT * y + μ_wT

    top_flux_inputs = standardize_wT.(top_flux_inputs)
    T_inputs = standardize_T.(T_inputs)
    wT_outputs = standardize_wT.(wT_outputs)

    n_data = length(top_flux_inputs)
    training_data = [((T_inputs[:, n], top_flux_inputs[n]), wT_outputs[:, n]) for n in 1:n_data] |> shuffle

    standardization = (
        T = (μ=μ_T, σ=σ_T, standardize=standardize_T, standardize⁻¹=standardize⁻¹_T),
        wT = (μ=μ_wT, σ=σ_wT, standardize=standardize_wT, standardize⁻¹=standardize⁻¹_wT)
    )

    return training_data, standardization
end

#=

function train_on_heat_flux!(loss, params, training_data, optimizer)
    function cb()
        μ_loss = mean(loss(input, wT) for (input, wT) in training_data)
        @info @sprintf("Training on heat flux... mean(loss) = %e", μ_loss)
        return loss
    end

    Flux.train!(loss, params, training_data, optimizer, cb=Flux.throttle(cb, 2))

    return nothing
end

function animate_learned_heat_flux(ds, NN, standardization; grid_points, filepath, frameskip=1, fps=15, npde=nothing)
    T, wT, zF = ds["T"], ds["wT"], ds["zF"]
    Nz, Nt = size(T)
    zF_coarse = coarse_grain(zF, grid_points+1, Face)

    Q  = nc_constant(ds, "Heat flux")
    ρ₀ = nc_constant(ds, "Reference density")
    cₚ = nc_constant(ds, "Specific_heat_capacity")
    top_flux = Q / (ρ₀ * cₚ)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $filepath [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(wT[:, n], zF, linewidth=2, xlim=(-1e-5, 3e-5), ylim=(-100, 0),
             label="Oceananigans wT", xlabel="Heat flux", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        S_T = standardization.T.standardize
        S_wT, S⁻¹_wT = standardization.wT.standardize, standardization.wT.standardize⁻¹

        if NN isa DiffEqFlux.FastChain
            T_NN = coarse_grain(T[:, n], grid_points, Cell) .|> S_T
            wT_NN = NN(T_NN, npde.p) .|> S⁻¹_wT
            plot!(wT_NN, zF_coarse, linewidth=2, label="Neural network")
        else
            input = (coarse_grain(T[:, n], grid_points, Cell) .|> S_T, top_flux .|> S_wT)
            wT_NN = input |> NN .|> S⁻¹_wT
            plot!(wT_NN, zF_coarse, linewidth=2, label="Neural network")
        end
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)

    return nothing
end

function construct_neural_pde(NN, ds, standardization; grid_points, iterations)
    H = abs(ds["zF"][1])
    τ = ds["time"][end]

    Nz = grid_points
    zC = coarse_grain(ds["zC"], Nz, Cell)
    Δẑ = diff(zC)[1] / H

    Dzᶠ = Dᶠ(Nz, Δẑ)

    # Set up neural network for non-dimensional PDE
    # ∂T/dt = - ∂z(wT) + ...
    σ_T, σ_wT = standardization.T.σ, standardization.wT.σ

    NN_∂T∂t = FastChain(NN.layers...,
                        (wT, _) -> - Dzᶠ * wT,
                        (∂z_wT, _) -> σ_wT/σ_T * τ/H * ∂z_wT)

    # Set up and return neural differential equation
    Nt = length(ds["time"])
    tspan = (0.0, maximum(iterations) / Nt)
    tsteps = range(tspan[1], tspan[2], length = length(iterations))

    return NeuralODE(NN_∂T∂t, tspan, ROCK4(), reltol=1e-3, saveat=tsteps)
end

function train_free_convection_neural_pde!(npde, loss, opt; maxiters)
    function cb(θ, μ_loss, loss_T, loss_∂T∂z)
        @info @sprintf("Training free convection neural PDE... loss = %e (loss_T = %e, loss_∂T∂z = %e)",
                       μ_loss, loss_T, loss_∂T∂z)
        return false
    end

    if opt isa Optim.AbstractOptimizer
        @info "Training free convection neural PDE with $(typeof(opt)).."
        res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=cb)
        display(res)
        npde.p .= res.minimizer
    else
        @info "Training free convection neural PDE with $(typeof(opt))(η=$(opt.eta))..."
        res = DiffEqFlux.sciml_train(loss, npde.p, opt, cb=cb, maxiters=maxiters)
        display(res)
        npde.p .= res.minimizer
    end

    return nothing
end

function animate_learned_free_convection(ds, npde, standardization; grid_points, iters, filepath, fps=15)
    T, wT, z = ds["T"], ds["wT"], ds["zC"]
    Nz, Nt = size(T)
    z_coarse = coarse_grain(z, grid_points, Cell)

    S_T, S⁻¹_T = standardization.T.standardize, standardization.T.standardize⁻¹

    T₀_NN = coarse_grain(T[:, 1], grid_points, Cell) .|> S_T
    sol_npde = npde(T₀_NN) |> Array

    time_steps = size(sol_npde, 2)
    anim = @animate for (i, n) in enumerate(iters)
        @info "Plotting $filepath [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(T[:, n], z, linewidth=2, xlim=(19, 20), ylim=(-100, 0),
             label="Oceananigans T(z,t)", xlabel="Temperature (°C)", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(S⁻¹_T.(sol_npde[:, i]), z_coarse, linewidth=2, label="Neural PDE")
    end

    @info "Saving $filepath"
    mp4(anim, filepath, fps=fps)

    return nothing
end

#####
##### Problem parameters
#####

Nz = 32  # Number of grid points for the neural PDE.

skip_first = 5  # Number of transient time snapshots to skip at the start
future_time_steps = 1

#####
##### Pick training and testing data
#####

Qs = (25, 50, 75, 100)
Qs_train = (25, 75)
Qs_test  = (50, 100)

#=

#####
##### Prepare heat flux training data
#####

training_data_heat_flux, standardization =
    free_convection_heat_flux_training_data(ds, Qs_train, grid_points=Nz, skip_first=skip_first)

n_obs = length(training_data_heat_flux)
@info "Heat flux training data contains $n_obs pairs."

data_loader_heat_flux = Flux.Data.DataLoader(training_data_heat_flux, batchsize=n_obs, shuffle=true)

n_obs = data_loader_heat_flux.nobs
batch_size = data_loader_heat_flux.batchsize
n_batches = ceil(Int, n_obs / batch_size)
@info "Heat flux data loader contains $n_obs observations (batch size = $batch_size)."

#####
##### Define temperature T -> heat flux wT neural network
#####

NN_heat_flux_filepath = "NN_heat_flux.bson"

if isfile(NN_heat_flux_filepath)
    @info "Loading $NN_heat_flux_filepath..."
    BSON.@load NN_heat_flux_filepath NN
else
    NN = Chain(Dense( Nz, 4Nz, relu),
               Dense(4Nz, 4Nz, relu),
               Dense(4Nz, Nz-1))
end

bot_flux = standardization.wT.standardize(0)

function NN_heat_flux(input)
    T, top_flux = input
    ϕ = NN(T)
    wT = cat(bot_flux, ϕ, top_flux, dims=1)
    return wT
end

#####
##### Train neural network on temperature T -> heat flux wT mapping
#####

if !isfile(NN_heat_flux_filepath)
    loss_heat_flux(input, wT) = Flux.mse(NN_heat_flux(input), wT)

    epochs = 1
    optimizers = [ADAM(1e-2)]

    for opt in optimizers, e in 1:epochs, (i, mini_batch) in enumerate(data_loader_heat_flux)
        @info "Training heat flux with $(typeof(opt))(η=$(opt.eta))... (epoch $e/$epochs, mini-batch $i/$n_batches)"
        train_on_heat_flux!(loss_heat_flux, Flux.params(NN), mini_batch, opt)
    end

    @info "Saving $NN_heat_flux_filepath..."
    BSON.@save NN_heat_flux_filepath NN

    for Q in Qs_train
        filepath = "learned_heat_flux_initial_guess_Q$(Q)W.mp4"
        animate_learned_heat_flux(ds[Q], NN_heat_flux, standardization, grid_points=Nz, filepath=filepath, frameskip=5, fps=15)
    end
end

#####
##### Preparing to train free convection T(z,t) neural PDE
#####

NN_fast = FastChain(NN)

function generate_NN_fast_heat_flux(NN, bottom_flux, top_flux)
    return FastChain(
        NN,
        (wT, _) -> cat(bottom_flux, wT, top_flux, dims=1)
    )
end

S_T  = standardization.T.standardize
S_wT = standardization.wT.standardize

ρ₀ = nc_constant(ds[75], "Reference density")
cₚ = nc_constant(ds[75], "Specific_heat_capacity")

flux_standarized(Q) = Q / (ρ₀ * cₚ) |> S_wT

best_weights, _ = Flux.destructure(NN)

H  = abs(ds[Qs[1]]["zF"][1])
zF = coarse_grain(ds[Qs[1]]["zF"], Nz, Face)
Δẑ = diff(zF)[2] / H

Dzᶜ = Dᶜ(Nz, Δẑ)

#####
##### Train on multiple simulations at once while incrementally increasing the time span
#####

training_intervals = (1:50, 1:100, 1:2:201, 1:4:401, 1:8:801, 1:9:length(ds[25]["time"]))
training_maxiters  = (50,   50,    100,     100,     100,     100)
training_epochs    = (1,    2,     2,       2,       2,       3)

training_intervals = [1:9:length(ds[25]["time"])]
training_maxiters  = [100]
training_epochs    = [10]

training_intervals = [1:9:length(ds[25]["time"])]
training_maxiters  = [500]
training_epochs    = [1]

for (iters_train, maxiters, epochs) in zip(training_intervals, training_maxiters, training_epochs), e in 1:epochs
    global best_weights

    training_data_time_step = cat([cat((coarse_grain(ds[Q]["T"][:, n], Nz, Cell) .|> S_T for n in iters_train)..., dims=2) for Q in Qs_train]..., dims=2)

    T₀s = Dict(Q => coarse_grain(ds[Q]["T"][:, iters_train[1]], Nz, Cell) .|> S_T for Q in Qs_train)

    NNs_fast_heat_flux = Dict(
        Q => generate_NN_fast_heat_flux(NN_fast, flux_standarized(0), flux_standarized(Q))
        for Q in Qs_train
    )

    npdes = Dict(
        Q => construct_neural_pde(NNs_fast_heat_flux[Q], ds[Q], standardization, grid_points=Nz, iterations=iters_train)
        for Q in Qs_train
    )

    for Q in Qs_train
        npdes[Q].p .= best_weights
    end

    function combined_loss(θ)
        sols_npde = cat([Array(npdes[Q](T₀s[Q], θ)) for Q in Qs_train]..., dims=2)
        dTdz = cat([Dzᶜ * sols_npde[:, n] for n in 1:size(sols_npde, 2)]..., dims=2)

        C = 5  # loss_dTdz will always be weighted with 0 <= weight <= C
        loss_T = Flux.mse(sols_npde, training_data_time_step)
        loss_dTdz = mean(min.(dTdz, 0) .^ 2)
        weighted_loss = loss_T + min(C * loss_T, loss_dTdz)

        return weighted_loss, loss_T, loss_dTdz
    end

    @info "Training free convection neural PDE for iterations $iters_train (epoch $e/$epochs)..."
    η = (epochs - e + 1) * 1e-3
    train_free_convection_neural_pde!(npdes[Qs_train[1]], combined_loss, ADAM(η), maxiters=maxiters)

    best_weights .= npdes[Qs_train[1]].p
end

npde_filename = "free_convection_neural_pde_parameters.bson"
@info "Saving $npde_filename..."
BSON.@save npde_filename best_weights

#####
##### Quantify testing and training errors
#####

for Q in (Qs_train..., Qs_test...)
    iters_train = training_intervals[end]
    sol_correct = cat((coarse_grain(ds[Q]["T"][:, n], Nz, Cell) .|> S_T for n in iters_train)..., dims=2)
    T₀ = coarse_grain(ds[Q]["T"][:, iters_train[1]], Nz, Cell) .|> S_T

    NN_fast_heat_flux = generate_NN_fast_heat_flux(NN_fast, flux_standarized(0), flux_standarized(Q))
    npde = construct_neural_pde(NN_fast_heat_flux, ds[Q], standardization, grid_points=Nz, iterations=iters_train)
    npde.p .= best_weights
    sol_npde = Array(npde(T₀, npde.p))

    μ_loss = Flux.mse(sol_npde, sol_correct)
    @info @sprintf("Q = %dW loss: %e", Q, μ_loss)
end

#####
##### Animate learned heat flux and free convection solutions on training and testing simulations
#####

for Q in Qs
    regime = Q in Qs_train ? "training" : "testing"

    iters_train = training_intervals[end]

    bot_flux_S = flux_standarized(0)
    top_flux_S = flux_standarized(Q)

    NN_fast_heat_flux = generate_NN_fast_heat_flux(NN_fast, bot_flux_S, top_flux_S)
    npde = construct_neural_pde(NN_fast_heat_flux, ds[Q], standardization, grid_points=Nz, iterations=iters_train)
    npde.p .= best_weights

    filepath = "free_convection_neural_pde_$(regime)_$(Q)W.mp4"
    animate_learned_free_convection(ds[Q], npde, standardization, grid_points=Nz, iters=iters_train, filepath=filepath)

    filepath = "learned_heat_flux_$(regime)_$(Q)W.mp4"
    animate_learned_heat_flux(ds[Q], FastChain(npde.model.layers[1:end-2]...), standardization, grid_points=Nz, filepath=filepath, frameskip=5, fps=15, npde=npde)
end
