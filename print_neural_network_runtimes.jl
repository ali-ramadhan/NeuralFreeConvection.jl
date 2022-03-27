using Printf
using Statistics
using JLD2
using Flux

function prettytime(t)
    t > 3600 && return @sprintf("%.3f hours", t / 3600)
    return @sprintf("%.3f seconds", t)
end

nn_archs = ["dense_default", "dense_wider", "dense_deeper", "conv_2", "conv_4"]
labels = ["dense (default)", "dense (wider)", "dense (deeper)", "convolutional (2)", "convolutional (4)"]

for method in ("trained_on_fluxes", "trained_on_timeseries")
    for (arch, label) in zip(nn_archs, labels)
        filepath = joinpath("$(method)_$(arch)", "neural_network_history_$method.jld2")
        file = jldopen(filepath, "r")

        NN = file["neural_network/1"]
        n_params = sum(length, params(NN))

        epochs = keys(file["runtime"]) |> length
        runtimes = [file["runtime/$e"] for e in 2:epochs]  # ignore precompilation time

        close(file)

        runtime_total = sum(runtimes) |> prettytime
        runtime_mean = mean(runtimes) |> prettytime
        runtime_median = median(runtimes) |> prettytime
        runtime_min, runtime_max = extrema(runtimes) .|> prettytime

        @info "$method, $arch ($n_params parameters): $epochs epochs in $runtime_total (min=$runtime_min, median=$runtime_median, mean=$runtime_mean, max=$runtime_max)"
    end
end
