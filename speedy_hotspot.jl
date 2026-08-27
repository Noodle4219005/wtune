# SpeedyWeather.jl convective-hotspot runner for the wtune autotuner.
#
# Runs a fast global simplified-physics forecast (PrimitiveWetModel) over the
# case window and writes the ACCUMULATED convective rain field as a JSON point
# list {points: [[lon,lat,val],...]}. Accumulated rain over the run == already
# time-integrated -> the convective load map, not a t=0 snapshot.
#
# FIDELITY: idealized initial conditions -> a dynamically-plausible convection
# pattern, NOT this case's exact storm. For case-accurate convection, initialize
# from analysis (ERA5). Use for the SHAPE/MAGNITUDE of convective imbalance.
#
# Usage: julia --project=jlenv speedy_hotspot.jl --truncation 31 --nlayers 8 --hours 6 --out out.json

using SpeedyWeather

function argval(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i+1]
end

truncation = parse(Int, argval("--truncation", argval("--trunc", "31")))
nlayers    = parse(Int, argval("--nlayers", "8"))
hours      = parse(Int, argval("--hours", "6"))
outpath    = argval("--out", "speedy_hotspot.json")

println("SpeedyWeather ", pkgversion(SpeedyWeather),
        " | truncation=", truncation, " nlayers=", nlayers, " hours=", hours)

spectral_grid = SpectralGrid(truncation=truncation, nlayers=nlayers)
model = PrimitiveWetModel(spectral_grid)
sim = initialize!(model)
run!(sim, period=Hour(hours))

rain = sim.variables.parameterizations.rain_convection    # accumulated conv rain [m]

# grid coordinates (degrees). Try the field's grid; fall back to spectral_grid.grid.
grid = try
    rain.grid
catch
    spectral_grid.grid
end

# get_londlatds returns a 2-tuple of vectors: (londs, latds), one entry per grid cell
londs, latds = RingGrids.get_londlatds(grid)

vals = try
    Array(rain)
catch
    rain.data
end

n = length(vals)
@assert length(londs) == n "coords ($(length(londs))) != values ($n)"

open(outpath, "w") do io
    print(io, "{\"provider\":\"speedyweather\",\"version\":\"", pkgversion(SpeedyWeather), "\",")
    print(io, "\"truncation\":", truncation, ",\"trunc\":", truncation, ",")
    print(io, "\"nlayers\":", nlayers, ",\"period_hours\":", hours, ",")
    print(io, "\"units\":\"m_accum_conv_rain\",")
    print(io, "\"note\":\"idealized IC (not case ERA5); dynamically-plausible convection, not this exact storm\",")
    print(io, "\"points\":[")
    for i in 1:n
        lo = Float64(londs[i]); la = Float64(latds[i])
        lo = lo > 180 ? lo - 360 : lo
        v = Float64(vals[i])
        print(io, "[", round(lo, digits=3), ",", round(la, digits=3), ",", round(v, digits=8), "]")
        i < n && print(io, ",")
    end
    print(io, "]}")
end

println("wrote ", outpath, " (", n, " points, max conv rain ", round(maximum(Float64.(vals)), digits=6), " m)")
