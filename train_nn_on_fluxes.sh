#!/bin/sh

julia --project train_nn_on_fluxes.jl --grid-points 32 --base-parameterization convective_adjustment --training-simulations 1 2 3 4 5 6 7 8 9 --epochs 5000 --neural-network-architecture dense-default --output-directory trained_on_fluxes_dense_default
julia --project train_nn_on_fluxes.jl --grid-points 32 --base-parameterization convective_adjustment --training-simulations 1 2 3 4 5 6 7 8 9 --epochs 5000 --neural-network-architecture dense-wider --output-directory trained_on_fluxes_dense_wider
julia --project train_nn_on_fluxes.jl --grid-points 32 --base-parameterization convective_adjustment --training-simulations 1 2 3 4 5 6 7 8 9 --epochs 5000 --neural-network-architecture dense-deeper --output-directory trained_on_fluxes_dense_deeper
julia --project train_nn_on_fluxes.jl --grid-points 32 --base-parameterization convective_adjustment --training-simulations 1 2 3 4 5 6 7 8 9 --epochs 5000 --neural-network-architecture conv-2 --output-directory trained_on_fluxes_conv_2
julia --project train_nn_on_fluxes.jl --grid-points 32 --base-parameterization convective_adjustment --training-simulations 1 2 3 4 5 6 7 8 9 --epochs 5000 --neural-network-architecture conv-4 --output-directory trained_on_fluxes_conv_4
