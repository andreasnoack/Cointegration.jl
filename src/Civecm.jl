# Min Civecm kode
module Civecm

using Stats
using Distributions

import Base: convert, copy, eigfact, eigvals, show, LinAlg.syrk_wrapper
import Stats: loglikelihood, residuals
# import Profile.@iprofile
export β, τ, ρ, δ, civecmI1, civecmI2, civecmI2alt, companion, loglikelihood, lrtest, ranktest, setrank, show, simulate, VAR

include("auxilliary.jl")
include("abstract.jl")
include("I1.jl")
include("I2ParuoloRahbek.jl")
include("VAR.jl")
# include("I2MosconiParuolo.jl")
# include("I2Givens.jl")

end