using Cointegration, Test, LinearAlgebra, StatsBase, Statistics, Random, CSV

@testset "Basic" begin
    k = 5
    n = 200

    rng = MersenneTwister(123)
    X = cumsum(randn(rng, n, k), dims = 1)

    @testset "I(1) model" begin
        f = civecmI1(X, lags=2)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), f) == """
Cointegration.CivecmI1

α:
5×5 Array{Float64,2}:
 -0.134816    0.0379236   0.0641901  -0.134372    -0.000172909
 -0.0325848  -0.143578   -0.158585   -0.0222283    0.00232374 
  0.249325   -0.132182    0.0625736  -0.055161     0.000511961
 -0.0791614  -0.136955    0.0874696   0.0310692    0.00433959 
 -0.0813468  -0.127097    0.0109304   0.00614004  -0.00688657 

βᵀ:
5×5 Array{Float64,2}:
 -0.111755   -0.0437334  -0.384134   -0.0352714    0.0854627
  0.0420461   0.119863   -0.107709    0.237534     0.0313933
 -0.134803    0.217288   -0.0614371   0.00497531   0.0735451
  0.151711    0.14491     0.0980868  -0.134664    -0.0920703
 -0.0658796  -0.0631864   0.0218884   0.0823652   -0.0364104

Π:
5×5 Array{Float64,2}:
 -0.0123664    0.00492849   0.0305751   0.0321635   0.00676763
  0.015457    -0.0536112    0.0355952  -0.0305595  -0.0169934 
 -0.0502585   -0.0211768   -0.0907807  -0.0324101   0.0268204 
 -0.00427522   0.0102802    0.0429286  -0.0331307  -0.00765043
  0.00365877  -0.00797672   0.0447176  -0.0286602  -0.0104528 """

        rt = ranktest(f)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), rt) == """

 Rank    Value  p-value
    0   48.286    0.000
    1   28.189    0.000
    2   13.548    0.000
    3    4.653    0.000
    4    0.015    0.000
"""
        f3 = setrank(f, 3)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), f3) == """
Cointegration.CivecmI1

α:
5×3 Array{Float64,2}:
 -0.134816    0.0379236   0.0641901
 -0.0325848  -0.143578   -0.158585 
  0.249325   -0.132182    0.0625736
 -0.0791614  -0.136955    0.0874696
 -0.0813468  -0.127097    0.0109304

βᵀ:
3×5 Array{Float64,2}:
 -0.111755   -0.0437334  -0.384134   -0.0352714   0.0854627
  0.0420461   0.119863   -0.107709    0.237534    0.0313933
 -0.134803    0.217288   -0.0614371   0.00497531  0.0735451

Π:
5×5 Array{Float64,2}:
  0.00800789   0.0243893    0.0437589   0.0140827  -0.00561029
  0.0189824   -0.0502433    0.0377246  -0.0337443  -0.0189553 
 -0.0418562   -0.0131511   -0.0853814  -0.0398804   0.0217603 
 -0.00870288   0.00605219   0.0397861  -0.0293042  -0.00463187
  0.00227357  -0.00930161   0.044266   -0.0272662  -0.0101382 """
    end

    @testset "I(2) model" begin
        f = civecmI2(X, lags=2)
        rt = ranktest(f)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), rt) == """
5×6 Array{Float64,2}:
 737.952  565.325  410.35   272.83    151.23     48.286    
   0.0    423.178  277.116  139.94     36.9591   28.1893   
   0.0      0.0    148.919   36.1275   19.2267   13.548    
   0.0      0.0      0.0     18.3722    8.68801   4.6533   
   0.0      0.0      0.0      0.0       3.60138   0.0149133"""

        rng = MersenneTwister(123)
        rtsim = ranktest(rng, f, 100)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), rtsim[2]) == """
5×6 Array{Float64,2}:
 0.0  0.0  0.0  0.0   0.0   0.36
 0.0  0.0  0.0  0.0   0.32  0.46
 0.0  0.0  0.0  0.13  0.57  0.61
 0.0  0.0  0.0  0.23  0.57  0.57
 0.0  0.0  0.0  0.0   0.26  0.92"""

        f11 = setrank(f, 1, 1)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), f11) == """
β':
1×5 Array{Float64,2}:
 -0.0220894  -0.00747066  -0.0701127  -0.01049  0.0135633

τ⊥δ:
5×1 Array{Float64,2}:
  0.899925 
  0.0519376
 -0.321931 
  0.0357782
 -0.142247 

τ':
2×5 Array{Float64,2}:
 -0.118063  -0.037522  -0.378734  -0.0603913  0.0813303
  0.179069   0.322215   0.133885  -0.385039   0.850675 """
    end
end

@testset "Illustrative examples from Johansen's book" begin
    danish_df  = CSV.read(joinpath(@__DIR__(), "..", "data", "danish.csv"))
    danish_mat = [danish_df.log_real_money danish_df.log_real_income danish_df.bond_rate danish_df.deposit_rate]

    # Create constant term and seasonals.
    cnst = [ones(55) repeat(Matrix{Float64}(I, 4, 3), 14)[1:55, :]]
    # Johansen makes the seasonals orthogonal to the constant term
    proj(v, u) = (v'u)/(u'u)*u
    for k in 2:4
        cnst[:, k] .-= sum(proj(cnst[:, k], cnst[:, j]) for j in 1:k-1)
    end

    f = civecmI1(danish_mat, unrestricted=cnst, lags=2)

    @testset "Table 2.1" begin
        @test f.Π[:,1:4] ≈ [-0.181  0.110 -1.042  0.638
                             0.186 -0.309  0.658 -0.648
                             0.014 -0.018  0.082 -0.167
                            -0.004  0.020  0.143 -0.314] atol=2e-3
    end

    @testset "Table 2.2" begin
        @test f.Γ[:,1:4] ≈ [0.195 -0.096 -0.138 -0.462
                            0.504 -0.045 -0.377  0.060
                            0.051  0.136  0.301  0.253
                            0.069 -0.022  0.227  0.265] atol=2e-3
    end

    @testset "Table 2.3" begin
        @testset "The constant" begin
            @test f.Γ[:, 5] ≈ [1.583, -0.390, -0.064, -0.071] atol=2e-3
        end

        @testset "The seasonal" begin
            @test_broken f.Γ[:, 6:8] ≈ [-0.023  0.016 -0.039
                                        -0.019 -0.007 -0.032
                                        -0.003 -0.007 -0.007
                                        -0.002  0.001 -0.003]
        end
    end

    @testset "Table 2.4" begin
        @test cor(residuals(f)) ≈ [ 1.00  0.53 -0.45 -0.31
                                    0.53  1.00 -0.08 -0.24
                                   -0.45 -0.08  1.00  0.25
                                   -0.31 -0.24  0.25  1.00] atol=1e-2
    end

    @testset "Table 2.5" begin
        # FIXME! ARCH and JB tests not implemented yet
        @test mapslices(skewness, residuals(f), dims=1) ≈ [0.552 0.524 -0.297 0.415] atol=1e-3
        @test mapslices(kurtosis, residuals(f), dims=1) ≈ [-0.075 -0.087 0.576 0.562] atol=1e-3
    end

    @testset "Table 2.6" begin
        @test sort(eigvals(f), by=real, rev=true) ≈ [ 0.9725,
                                                     0.7552 - 0.1571im,
                                                     0.7552 + 0.1571im,
                                                     0.6051,
                                                     0.5955 - 0.3143im,
                                                     0.5955 + 0.3143im,
                                                    -0.1425 - 0.2312im,
                                                    -0.1425 + 0.2312im] atol=1e-3
    end
end