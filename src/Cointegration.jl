module Cointegration

using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Random
using Missings

using Distributions

import Base: convert, copy, show
import LinearAlgebra: eigen, eigvals, syrk_wrapper!
import StatsBase: loglikelihood, residuals
import Distributions: estimate
import ProgressMeter: Progress, next!

export β,
    τ,
    ρ,
    δ,
    VAR,
    bootstrap,
    civecmI1,
    civecmI2,
    civecmI2alt,
    companion,
    estimate,
    lrtest,
    normalitytest,
    ranktest,
    restrict,
    setrank,
    simulate
# From StatsBase
export residuals

include("auxiliary.jl")
include("abstract.jl")
include("I1.jl")
include("I2ParuoloRahbek.jl")
include("VAR.jl")
# include("I2MosconiParuolo.jl")
include("I2Givens.jl")

end
