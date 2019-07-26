PIPS.jl
========

**WIP**

**PIPS.jl** is a [Julia](http://julialang.org/) interface to the [PIPS](https://github.com/Argonne-National-Laboratory/PIPS) nonlinear solver. This interface matches [StructJuMP](https://github.com/StructJuMP/StructJuMP.jl)'s parallel AML capabilities for block structured optimization problems.

This interface only supports [JuMP](https://github.com/JuliaOpt/JuMP.jl) via [MathOptInterface](https://github.com/JuliaOpt/MathOptInterface.jl) and was created specifically to move away from `MathProgBase`.

## Installation

In the final release this package will be registered in `METADATA.jl` and so can be installed with `Pkg.add`.

```
julia> import Pkg; Pkg.add("Pips")
```

`PIPS.jl` requires the user to build PIPS manually. Information on this can be found its [webpage](https://github.com/Argonne-National-Laboratory/PIPS) or please contact one of the developers.
