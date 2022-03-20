function ConvectiveAdjustmentNDE(NN, ds; iterations=nothing)
    weights, reconstruct = Flux.destructure(NN)

    T = ds["T"]
    wT = ds["wT"]
    FT = eltype(T)
    Nz = size(T, 3)
    Nt = size(T, 4)
    zc = znodes(T)
    zf = znodes(wT)
    times = T.times

    H = abs(zf[1]) # Domain depth/height
    τ = times[end]  # Simulation length

    Δẑ = diff(zc)[1] / H  # Non-dimensional grid spacing
    Dzᶜ = Dᶜ(Nz, Δẑ) # Differentiation matrix operator
    Dzᶠ = Dᶠ(Nz, Δẑ) # Differentiation matrix operator
    Dzᶜ = convert(Array{FT}, Dzᶜ)
    Dzᶠ = convert(Array{FT}, Dzᶠ)

    if isnothing(iterations)
        iterations = 1:length(times)
    end

    """
    Non-dimensional PDE is

        ∂T/∂t = - σ_wT/σ_T * τ/H * ∂/∂z(wT - K ∂T/∂z)

    where K = 0 if ∂T/∂z < 0 and K = K_CA if ∂T/∂z > 0.
    """
    function ∂T∂t(T, p, t)
        weights = p[1:end-7]
        bottom_flux, top_flux, σ_T, σ_wT, H, τ, K_CA = p[end-6:end]

        # Turbulent heat flux
        NN = reconstruct(weights)
        wT_interior = NN(T)
        wT = [bottom_flux; wT_interior; top_flux]
        ∂z_wT = Dzᶜ * wT

        # Convective adjustment
        ∂T∂z = Dzᶠ * T
        ∂z_K∂T∂z = Dzᶜ * min.(0, K_CA * ∂T∂z)

        return σ_wT/σ_T * τ/H * (- ∂z_wT .+ ∂z_K∂T∂z)
    end

    tspan = FT.( (0.0, maximum(iterations) / Nt) )
    saveat = range(tspan[1], tspan[2], length=length(iterations))

    # See: https://github.com/SciML/DiffEqFlux.jl/blob/449efcecfc11f1eab65d0e467cf57db9f5a5dbec/src/neural_de.jl#L66-L67
    # We set the initial condition to `nothing`. Then we will set it to some actual initial condition when calling `solve`.
    ff = ODEFunction{false}(∂T∂t, tgrad=DiffEqFlux.basic_tgrad)
    return ODEProblem{false}(ff, nothing, tspan, saveat=saveat)
end

function ConvectiveAdjustmentNDEParameters(ds, T_scaling, wT_scaling, K_CA)
    wT = ds["wT"]
    zf = znodes(wT)
    times = wT.times

    H = abs(zf[1]) # Domain depth/height
    τ = times[end]  # Simulation length

    bottom_flux = wT_scaling(interior(wT)[1, 1, 1, 1])
    top_flux = wT_scaling(ds.metadata["temperature_flux"])

    FT = eltype(wT)
    return FT.([bottom_flux, top_flux, T_scaling.σ, wT_scaling.σ, H, τ, K_CA])
end
