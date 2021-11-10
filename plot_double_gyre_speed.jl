using Printf
using NCDatasets
using CairoMakie

ds = NCDataset("double_gyre.nc")

xc = ds["xC"] / 1000
yc = ds["yC"] / 1000
zc = ds["zC"] / 1000
xf = ds["xF"] / 1000
yf = ds["yF"] / 1000
zf = ds["zF"] / 1000

Nx, Ny, Nz = length(xc), length(yc), length(zc)

Nx½ = round(Int, Nx/2)
Ny½ = round(Int, Ny/2)

for var in ["speed"]# ("u", "v", "T")
    frame = Node(1)
    title = @lift "Double gyre day $(round(Int, ds["time"][$frame] / 86400))"
    var_surface = @lift ds[var][:, :, Nz, $frame]
    var_meridional = @lift ds[var][:, Ny½, :, $frame]
    var_zonal = @lift ds[var][Nx½, :, :, $frame]

    cmap = var == "T" ? :thermal : :matter
    clims = var == "T" ? (0, 35) : (0, 1)
    label = var == "T" ? "temperature (°C)" : "speed (m/s)"

    fig = Figure(resolution=(1920, 1080))

    ax1 = fig[1, 1] = Axis(fig, title="$var(x, y, 0)", xlabel="x (km)", ylabel="y (km)")
    hm1 = heatmap!(ax1, xc, yc, var_surface, colormap=cmap, colorrange=clims)
    # xlims!(ax1, extrema(xf))
    # ylims!(ax1, extrema(yf))

    #=
    ax2 = fig[1, 2] = Axis(fig, title="$var(x, 0, z)", xlabel="x (km)", ylabel="z (km)")
    hm2 = heatmap!(ax2, xc, zc, var_meridional, colormap=cmap, colorrange=clims)

    ax3 = fig[1, 3] = Axis(fig, title="$var(0, y, z)", xlabel="y (km)", ylabel="z (km)")
    hm3 = heatmap!(ax3, yc, zc, var_zonal, colormap=cmap, colorrange=clims)
    =#

    cb1 = fig[1, 2] = Colorbar(fig, hm1, label=label, width=30)

    supertitle = fig[0, :] = Label(fig, title, textsize=30)

    record(fig, "double_gyre_$var.mp4", 1:length(ds["time"]), framerate=30) do n
        @info "Animating double gyre $var frame $n/$(length(ds["time"]))..."
        frame[] = n
    end
end

Nxs = round.(Int, [Nx/10, Nx/2, 9Nx/10])
Nys = round.(Int, [Ny/10, Ny/2, 9Ny/10])

frame = Node(1)
fig = Figure(resolution = (1920, 1080))

for (i, nx) in enumerate(Nxs), (j, ny) in enumerate(Nys)
    title = @sprintf("x = %d km, y = %d km", xc[nx], yc[ny])
    T_profile = @lift ds["T"][nx, ny, :, $frame]

    ax = fig[i, j] = Axis(fig, title=title, xlabel="Temperature (°C)", ylabel="z (km)")
    T_plot = lines!(ax, T_profile, zc, linewidth=3)

    xlims!(ax, (0, 35))
    ylims!(ax, extrema(zf))
end

title = @lift "Double gyre day $(round(Int, ds["time"][$frame] / 86400))"
supertitle = fig[0, :] = Label(fig, title, textsize=30)

record(fig, "double_gyre_T_profiles.mp4", 1:length(ds["time"]), framerate=30) do n
    @info "Animating double gyre T profiles frame $n/$(length(ds["time"]))..."
    frame[] = n
end

close(ds)
