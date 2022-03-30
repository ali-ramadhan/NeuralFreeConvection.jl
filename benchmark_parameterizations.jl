using Statistics
using Printf
using JLD2
using Flux
using BenchmarkTools
using OceanTurb
using Oceananigans
using FreeConvection

function run_kpp_simulation(N, H, T₀, FT, ∂T∂z, times, constants, parameters; Δt=600)
    model = OceanTurb.KPP.Model(N=N, H=H, stepper=:BackwardEuler, constants=constants, parameters=parameters)

    model.solution.T.data[1:N] .= T₀

    model.bcs.T.top = OceanTurb.FluxBoundaryCondition(FT)
    model.bcs.T.bottom = OceanTurb.GradientBoundaryCondition(∂T∂z)

    for t in times
        OceanTurb.run_until!(model, Δt, t)
    end

    return
end

function benchmark_kpp_simulation(ds)
    ρ₀ = 1027.0
    cₚ = 4000.0
    f  = ds.metadata["coriolis_parameter"]
    α  = ds.metadata["thermal_expansion_coefficient"]
    β  = 0.0
    g  = ds.metadata["gravitational_acceleration"]
    constants = OceanTurb.Constants(Float64, ρ₀=ρ₀, cP=cₚ, f=f, α=α, β=β, g=g)
    parameters = OceanTurb.KPP.Parameters(CSL=0.1, CNL=4.0, Cb_T=0.5, CKE=1.5)

    T = ds["T"]
    wT = ds["wT"]
    Nz = size(T, 3)
    zf = znodes(wT)
    H = abs(zf[1])
    times = T.times

    T₀ = interior(T)[1, 1, :, 1]

    FT = ds.metadata["temperature_flux"]
    ∂T∂z = ds.metadata["dθdz_deep"]

    b = @benchmark run_kpp_simulation($Nz, $H, $T₀, $FT, $∂T∂z, $times, $constants, $parameters)

    return b
end

output_dir = "trained_on_fluxes_dense_default"
nn_filepath = joinpath(output_dir, "neural_network_trained_on_fluxes.jld2")

file = jldopen(nn_filepath, "r")
T_scaling = file["T_scaling"]
wT_scaling = file["wT_scaling"]
close(file)

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets
