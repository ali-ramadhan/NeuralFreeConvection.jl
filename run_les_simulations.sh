#!/bin/sh

# small thermocline
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 1e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 48 --thermocline-width 24 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 1e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_1
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 3e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 48 --thermocline-width 24 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 1e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_2
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 5e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 48 --thermocline-width 24 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 1e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_3

# medium thermocline
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 1e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 48 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 1.5e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_4
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 3e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 48 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 1.5e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_5
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 5e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 48 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 1.5e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_6

# large thermocline
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 1e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 64 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 2e-5 --deep-buoyancy-gradient 5e-6 --name free_convection_7
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 3e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 64 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 2e-5 --deep-buoyancy-gradient 5e-6 --name free_convection_8
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 5e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 64 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 2e-5 --deep-buoyancy-gradient 5e-6 --name free_convection_9

# testing for Qb interpolation
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 4e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 48 --thermocline-width 24 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 1e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_10
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 2e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 48 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 1.5e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_11
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 4e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 64 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 2e-5 --deep-buoyancy-gradient 5e-6 --name free_convection_12

# testing for Qb extrapolation
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 6e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 48 --thermocline-width 24 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 1e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_13
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 0.5e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 48 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 1.5e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_14
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 6e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 64 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 2e-5 --deep-buoyancy-gradient 5e-6 --name free_convection_15

# testing for N² interpolation
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 1e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 36 --thermocline-width 36 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 1.25e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_16
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 5e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 36 --thermocline-width 36 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 1.25e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_17
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 3e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 56 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 1.75e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_18

# testing for N² extrapolation
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 1e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 36 --thermocline-width 36 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 0.75e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_19
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 5e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 36 --thermocline-width 36 --surface-layer-buoyancy-gradient 2e-6 --thermocline-buoyancy-gradient 0.75e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_20
julia --project three_layer_constant_fluxes.jl --size 256 256 128 --hours 192 --buoyancy-flux 3e-8 --momentum-flux 0 --thermocline cubic --surface-layer-depth 24 --thermocline-width 56 --surface-layer-buoyancy-gradient 1e-6 --thermocline-buoyancy-gradient 2.25e-5 --deep-buoyancy-gradient 2e-6 --name free_convection_21
