using Statistics
using ArgParse
using JLD2
using Flux
using ColorTypes
using ColorSchemes
using CairoMakie

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--history"
            help = "Filepath to JLD2 file containing neural network history."
            arg_type = String
    end

    return parse_args(settings)
end

function movie3_neural_network_weights(history_filepath; filepath)
    file = jldopen(history_filepath)

    fig = Figure()

    xticksvisible = false
    yticksvisible = false
    xticklabelsvisible = false
    yticklabelsvisible = false
    xgridvisible = false
    ygridvisible = false
    leftspinevisible = false
    rightspinevisible = false
    topspinevisible = false
    bottomspinevisible = false
    ax_kwargs = (; xticksvisible, yticksvisible, xticklabelsvisible, yticklabelsvisible, xgridvisible, ygridvisible,
                   leftspinevisible, rightspinevisible, topspinevisible, bottomspinevisible)

    epoch = Node(1)
    NN = file["neural_network/$epoch"]

    spacing_weight_bias = 5

    for (l, layer) in enumerate(NN.layers)
        weight_size = size(layer.weight)
        bias_length = length(layer.bias)

        elements_x = weight_size[1] + spacing_weight_bias + 1
        elements_y = max(weight_size[2], bias_length)
        parameters = fill(NaN, (elements_x, elements_y))

        parameters[1:weight_size[1], 1:weight_size[2]] .= layer.weight
        parameters[end, 1:bias_length] .= layer.bias

        ax = Axis(fig[1, l], title="layer $l"; ax_kwargs...)
        heatmap!(ax, parameters, colormap=:curl, colorrange=(-0.5, 0.5))
    end

    Colorbar(fig[1, end+1], colormap=:curl, limits=(-0.5, 0.5))
    Label(fig[0, :], "Epoch 500")

    epochs = 100
    record(fig, filepath, 1:epochs; framerate=fps) do n
        @info "Animating $filepath frame $n/$Nt..."
        epoch[] = n
    end

    close(file)

    return fig
end

args = parse_command_line_arguments()
history_filepath = args["history"]
