# Min Civecm kode
using Stats
using Distributions

module Civecm

import Base: convert, copy, eigfact, eigvals, show, LinAlg.syrk_wrapper
import Stats: loglikelihood, residuals
export β, τ, ρ, δ, bootstrap, civecmI1, civecmI2, civecmI2alt, companion, estimate, lrtest, ranktest, setrank, show, simulate, VAR

include("auxilliary.jl")
include("abstract.jl")
include("I1.jl")
include("I2ParuoloRahbek.jl")
include("VAR.jl")
# include("I2MosconiParuolo.jl")
# include("I2Givens.jl")

end