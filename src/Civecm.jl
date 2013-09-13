using Stats
# Min Civecm kode
module Civecm

import Base: convert, copy, eigvals, show, LinAlg.syrk_wrapper
import Stats: loglikelihood, residuals
# import Profile.@iprofile
export β, τ, ρ, δ, civecmI1, civecmI2, civecmI2alt, loglikelihood, lrtest, ranktest, setrank, show, simulate, VAR

include("auxilliary.jl")
include("abstract.jl")
include("I1.jl")
include("I2ParuoloRahbek.jl")
include("VAR.jl")
# include("I2MosconiParuolo.jl")
# include("I2Givens.jl")

end