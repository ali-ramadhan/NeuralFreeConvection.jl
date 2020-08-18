using Printf

using NCDatasets
using Plots
using Flux

using ClimateSurrogates
using Oceananigans.Utils

include("free_convection_npde.jl")

function animate_variable(ds, var; grid_points, xlabel, xlim, frameskip, fps)
    Nz, Nt = size(ds[var])

    z_fine = ds["zC"]
    z_coarse = coarse_grain(ds["zC"], grid_points)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting $var [$n/$Nt]..."
        var_fine = ds[var][:, n]
        var_coarse = coarse_grain(ds[var][:, n], grid_points)

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(var_fine, z_fine, linewidth=2, xlim=xlim, ylim=(-100, 0),
             label="fine (Nz=$Nz)", xlabel=xlabel, ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        plot!(var_coarse, z_coarse, linewidth=2, label="coarse (Nz=$grid_points)")
    end

    filename = "free_convection_$var.mp4"
    @info "Saving $filename"
    mp4(anim, filename, fps=fps)

    return nothing
end

function free_convection_heat_flux_training_data(ds; grid_points)
    T, wT = ds["T"], ds["wT"]
    Nz, Nt = size(T)

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    training_data =
        [(coarse_grain( T[:, n], grid_points),
          coarse_grain(wT[:, n], grid_points))
         for n in 1:Nt]

    return training_data
end

function free_convection_time_step_training_data(ds; grid_points, future_time_steps=1)
    T = ds["T"]
    Nz, Nt = size(T)

    isinteger(Nz / grid_points) ||
        error("grid_points=$grid_points does not evenly divide Nz=$Nz")

    if future_time_steps == 1
        training_data =
            [(coarse_grain(T[:, n],   grid_points),
              coarse_grain(T[:, n+1], grid_points))
             for n in 1:Nt-1]
    else
        training_data =
            [(coarse_grain(T[:, n], grid_points),
              cat((coarse_grain(T[:, k], grid_points) for k in n+1:n+future_time_steps)..., dims=2))
             for n in 1:Nt-future_time_steps]
    end

    return training_data
end

function train_on_heat_flux!(NN, training_data, optimizer; epochs=1)
    loss(T, wT) = Flux.mse(NN(T), wT)

    function cb()
        Σloss = [loss(T, wT) for (T, wT) in training_data] |> sum
        @info "Training on heat flux... Σloss = $Σloss"
        return loss
    end

    for e in 1:epochs
        @info "Training on heat flux [epoch $e/$epochs]..."
        Flux.train!(loss, Flux.params(NN), training_data, optimizer)
        cb()
    end

    return nothing
end

function animate_learned_heat_flux(ds, NN; grid_points, frameskip, fps)
    T, wT, z = ds["T"], ds["wT"], ds["zC"]
    Nz, Nt = size(T)

    z_coarse = coarse_grain(z, grid_points)

    anim = @animate for n=1:frameskip:Nt
        @info "Plotting learned heat flux [$n/$Nt]..."

        time_str = @sprintf("%.2f days", ds["time"][n] / day)

        plot(wT[:, n], z, linewidth=2, xlim=(-1e-5, 3e-5), ylim=(-100, 0),
             label="Oceananigans wT", xlabel="Heat flux", ylabel="Depth z (meters)",
             title="Free convection: $time_str", legend=:bottomright, show=false)

        wT_NN = coarse_grain(T[:, n], grid_points) |> NN

        plot!(wT_NN, z_coarse, linewidth=2, label="Neural network")
    end

    filename = "free_convection_learned_heat_flux.mp4"
    @info "Saving $filename"
    mp4(anim, filename, fps=fps)

    return nothing
end

ds = NCDataset("free_convection_horizontal_averages.nc")

# Should not have saved constant units as strings...
nc_constant(ds, attr) = parse(Float64, ds.attrib[attr] |> split |> first)

Q = nc_constant(ds, "Heat flux")
ρ₀ = nc_constant(ds, "Reference density")
cₚ = nc_constant(ds, "Specific_heat_capacity")

surface_heat_flux = Q / (ρ₀ * cₚ)

Nz = 16  # Number of grid points for the neural PDE.

future_time_steps = 1

# animate_variable(ds, "T", grid_points=16, xlabel="Temperature T (°C)", xlim=(19, 20), frameskip=5, fps=15)
# animate_variable(ds, "wT", grid_points=16, xlabel="Heat flux", xlim=(-1e-5, 3e-5), frameskip=5, fps=15)

training_data_heat_flux = free_convection_heat_flux_training_data(ds, grid_points=Nz)
training_data_time_step = free_convection_time_step_training_data(ds, grid_points=Nz, future_time_steps=future_time_steps)

NN = free_convection_neural_pde_architecture(Nz, top_flux=surface_heat_flux, bottom_flux=0.0)

# train_on_heat_flux!(NN, training_data_heat_flux, Descent(1e-6), epochs=10)
# animate_learned_heat_flux(ds, NN, grid_points=Nz, frameskip=5, fps=15)
