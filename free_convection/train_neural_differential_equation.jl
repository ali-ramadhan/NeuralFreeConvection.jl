using Printf
using Statistics

using BSON
using Flux
using DifferentialEquations
using DiffEqFlux
using NCDatasets
using ClimateParameterizations

using Oceananigans.Grids: Cell, Face

using DiffEqSensitivity: InterpolatingAdjoint, ZygoteVJP

#####
##### Neural differential equation parameters
#####

Nz = 32  # Number of grid points

#####
##### Load weights and feature scaling from train_neural_network.jl
#####

neural_network_parameters = BSON.load("free_convection_neural_network_parameters.bson")

NN = neural_network_parameters[:weights]
T_scaling = neural_network_parameters[:T_scaling]
wT_scaling = neural_network_parameters[:wT_scaling]

#####
##### Load training data
#####

# Choose which free convection simulations to train on.
Qs_train = [25, 75]

# Load NetCDF data for each simulation.
ds = Dict(Q => NCDataset("free_convection_horizontal_averages_$(Q)W.nc") for Q in Qs_train)

#####
##### Utils for constructing the neural differential equation
#####

function FreeConvectionNDE(NN, ds, Nz, iterations)
    weights, reconstruct = Flux.destructure(NN)
    
    H = abs(ds["zF"][1]) # Domain height
    τ = ds["time"][end]  # Simulation length
    zC = coarse_grain(ds["zC"], Nz, Cell)
    Δẑ = diff(zC)[1] / H  # Non-dimensional grid spacing
    Dzᶠ = Dᶠ(Nz, Δẑ) # Differentiation matrix operator

    """
    Non-dimensional PDE is

        ∂T/∂t = - σ_wT/σ_T * τ/H * ∂/∂z(wT)
    """
    function ∂T∂t(T, p, t)
        weights = p[1:end-6]
        bottom_flux, top_flux, σ_T, σ_wT, H, τ = p[end-5:end]

        NN = reconstruct(weights)
        wT_interior = NN(T)
        wT = [bottom_flux; wT_interior; top_flux]
        ∂z_wT = Dzᶠ * σ_wT/σ_T * τ/H * wT
        return -∂z_wT
    end

    Nt = length(ds["time"])
    tspan = (0.0, maximum(iterations) / Nt)
    saveat = range(tspan[1], tspan[2], length = length(iterations))

    # We set the initial condition to `nothing` as we set it when
    # we call `solve`.
    return ODEProblem(∂T∂t, nothing, tspan, saveat=saveat)
end

function FreeConvectionNDEParameters(ds, T_scaling, wT_scaling)
    H = abs(ds["zF"][1]) # Domain height
    τ = ds["time"][end]  # Simulation length
    
    Q  = nc_constant(ds.attrib["Heat flux"])
    ρ₀ = nc_constant(ds.attrib["Reference density"])
    cₚ = nc_constant(ds.attrib["Specific_heat_capacity"])
    
    bottom_flux = wT_scaling(0)
    top_flux = wT_scaling(Q / (ρ₀ * cₚ))
    
    fixed_params = [bottom_flux, top_flux, T_scaling.σ, wT_scaling.σ, H, τ]
end

function initial_condition(ds, scaling=identity)
    T₀ = ds["T"][:, 1]
    T₀ = coarse_grain(T₀, Nz, Cell)
    return scaling.(T₀)
end

iterations = 1:100
nde = FreeConvectionNDE(NN, ds[75], Nz, iterations)
nde_params = FreeConvectionNDEParameters(ds[75], T_scaling, wT_scaling)
T₀ = initial_condition(ds[75], T_scaling)

#####
##### Train neural differential equation
#####

nde_training_data(ds, iterations, T_scaling) =
    cat([T_scaling.(coarse_grain(ds["T"][:, i], Nz, Cell)) for i in iterations]..., dims=2)

function train_free_convection_nde!(NN, nde, ds, iterations, T_scaling, epochs, opt)
    true_sol = nde_training_data(ds, iterations, T_scaling)

    function nde_loss()
        nn_weights, _ = Flux.destructure(NN)
        nde_sol = solve(nde, Tsit5(), u0=T₀, p=[nn_weights; nde_params],
                        sense=InterpolatingAdjoint(autojacvec=ZygoteVJP())) |> Array
        return Flux.mse(nde_sol, true_sol)
    end

    function nde_callback()
        @info @sprintf("Training free convection neural differential equation... loss = %.12e", nde_loss())
        return nothing
    end

    nn_params = Flux.params(NN)
    Flux.train!(nde_loss, nn_params, Iterators.repeated((), epochs), opt, cb=nde_callback)
end

for opt in [ADAM(), Descent()]
    train_free_convection_nde!(NN, nde, ds[75], iterations, T_scaling, 10, opt)
end

#=

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
