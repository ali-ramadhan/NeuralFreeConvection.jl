#!/bin/sh

julia --project train_free_convection_nde.jl --grid-points 32 --base-parameterization convective_adjustment --time-stepper ROCK4 --training-simulations 1 2 3 4 5 6 7 8 9 --training-epochs 500 --neural-network-architecture dense-default --output-directory free_convection_nde_dense_default
julia --project train_free_convection_nde.jl --grid-points 32 --base-parameterization convective_adjustment --time-stepper ROCK4 --training-simulations 1 2 3 4 5 6 7 8 9 --training-epochs 500 --neural-network-architecture dense-wider --output-directory free_convection_nde_dense_wider
julia --project train_free_convection_nde.jl --grid-points 32 --base-parameterization convective_adjustment --time-stepper ROCK4 --training-simulations 1 2 3 4 5 6 7 8 9 --training-epochs 500 --neural-network-architecture dense-deeper --output-directory free_convection_nde_dense_deeper
julia --project train_free_convection_nde.jl --grid-points 32 --base-parameterization convective_adjustment --time-stepper ROCK4 --training-simulations 1 2 3 4 5 6 7 8 9 --training-epochs 500 --neural-network-architecture conv-2 --output-directory free_convection_nde_conv_2
julia --project train_free_convection_nde.jl --grid-points 32 --base-parameterization convective_adjustment --time-stepper ROCK4 --training-simulations 1 2 3 4 5 6 7 8 9 --training-epochs 500 --neural-network-architecture conv-4 --output-directory free_convection_nde_conv_4