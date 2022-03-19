using Statistics
using Printf

using JLD2
using BlackBoxOptim
using OceanTurb
using FreeConvection

using Flux.Losses: mse

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

kpp_parameters = OceanTurb.KPP.Parameters(CSL=0.1, CNL=4.0, Cb_T=0.5, CKE=1.5)
kpp_solutions = Dict(id => free_convection_kpp(ds, parameters=kpp_parameters) for (id, ds) in data.coarse_training_datasets)

function kpp_one_simulation(parameters)
    CSL, CNL, Cb_T, CKE = parameters
    kpp_parameters = OceanTurb.KPP.Parameters(; CSL, CNL, Cb_T, CKE)
    kpp_solution = free_convection_kpp(data.coarse_training_datasets[1], parameters=kpp_parameters)

    T_LES = data.coarse_training_datasets[1]["T"][1, 1, 1:Nz, :] |> Array
    T_KPP = kpp_solution.T
    loss = (T_LES .- T_KPP) .^2 |> mean

    @info "Parameters = $parameters, loss = $loss"

    return loss
end

les_solutions = [ds["T"][1, 1, 1:Nz, :] |> Array for (id, ds) in data.coarse_training_datasets]

function kpp_training_simulations(parameters)
    CSL, CNL, Cb_T, CKE = parameters
    kpp_parameters = OceanTurb.KPP.Parameters(; CSL, CNL, Cb_T, CKE)

    kpp_solutions = [free_convection_kpp(ds, parameters=kpp_parameters) for (id, ds) in data.coarse_training_datasets]

    loss = mean(mse(kpp_solutions[id].T, les_solutions[id]) for id in 1:9)

    @info @sprintf("Parameters = %s, loss = %.3e\n", parameters, loss)

    return loss
end

# initial_guess = [0.1, 4.0, 0.5, 1.5]
initial_guess = [1.0, 4.0, 0.5, 3.0]
search_range = [(0.0, 1.0), (0.0, 8.0), (0.0, 8.0), (0.0, 8.0)]
res = bboptimize(kpp_training_simulations, initial_guess; SearchRange=search_range, NumDimensions=4, MaxTime=16*3600)

@show best_candidate(res)
@show best_fitness(res)

initial_fitness = kpp_training_simulations(initial_guess)
@info "best_fitness/initial_fitness = $initial_fitness/$(best_fitness(res)) = $(initial_fitness/best_fitness(res))"

jldopen("kpp_optimal_parameters.jld2", "w") do file
    file["result"] = res
end
