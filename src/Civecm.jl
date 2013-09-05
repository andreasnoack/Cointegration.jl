using Stats
# Min Civecm kode
module Civecm

import Base: convert, copy, eigvals, show, LinAlg.syrk_wrapper
import Stats: loglikelihood, residuals
# import Profile.@iprofile
export civecmI1, civecmI2, civecmI2alt, loglikelihood, lrtest, ranktest, setrank, show, simulate, VAR

include("auxilliary.jl")
include("abstract.jl")
include("I1.jl")
include("VAR.jl")
include("I2.jl")
# include("I2Givens.jl")

end