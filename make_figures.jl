using ArgParse

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--output-directory"
            help = "Output directory filepath."
            arg_type = String
    end

    return parse_args(settings)
end

include("figure5_loss.jl")
