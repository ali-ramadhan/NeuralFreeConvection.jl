using DiffEqFlux
using DiffEqFlux: FastLayer

"""
    ConservativeFluxLayer{T, B}

A neural network layer for imposing a specific flux at the top and bottom of a column model for some physical quantity.
"""
struct ConservativeFluxLayer{T, B}
              N :: Int
       top_flux :: T
    bottom_flux :: B

    function ConservativeFluxLayer(N; top_flux, bottom_flux)
        return new{typeof(top_flux), typeof(bottom_flux)}(N, top_flux, bottom_flux)
    end
end

(L::ConservativeFluxLayer)(ϕ, p...) = [L.bottom_flux, ϕ..., L.top_flux]

function free_convection_neural_pde_architecture(N; top_flux, bottom_flux)
    return Chain(Dense(N, 4N),
                 Dense(4N, N))
end
