using Cointegration, Test, LinearAlgebra, StatsBase, Statistics, StableRNGs, CSV, DataFrames, ZipFile, Dates

@testset "Basic" begin
    k = 5
    n = 200

    rng = StableRNG(123)
    X = cumsum(randn(rng, n, k), dims = 1)

    @testset "I(1) model" begin
        f = civecmI1(X, lags = 2)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), f) == """
Cointegration.CivecmI1

α:
5×5 Matrix{Float64}:
 -0.0947422   0.0761861   0.0257547  -0.0974986    0.0115806
  0.0632277   0.0336236   0.0637774   0.0303369    0.0670139
 -0.0428606  -0.0861965  -0.142879   -0.015907     0.032662
  0.165899    0.146969   -0.065865   -0.00883288  -0.0134935
  0.168351   -0.118823    0.0502608  -0.0596831    0.00882035

βᵀ:
5×5 Matrix{Float64}:
 0.150191    0.0384534   0.11206     0.0451019   -0.26955
 0.014997   -0.0768655   0.265054   -0.189269     0.268929
 0.0760236   0.145551    0.067286    0.00991004  -0.123557
 0.110725   -0.092395   -0.0527418   0.0568719    0.14021
 0.0712217  -0.0709114  -0.0881567  -0.0471977   -0.205761

Π:
5×5 Matrix{Float64}:
 -0.0210996   0.00243657   0.0154309   -0.024529     0.0267911
  0.0229809   0.00157467   0.0127809   -0.00431776  -0.0254161
 -0.0180272  -0.0166653   -0.0393039    0.010519    -0.0029248
  0.0201742  -0.0127313    0.0547689   -0.0208525    0.00448221
  0.0213436   0.0278115   -0.00687702   0.0267698   -0.093727"""

        normtest = normalitytest(f)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), normtest) == """
Cointegration.NormalityTest

Univarite tests:
Test values   df   p-values
       0.68    2       0.71
       4.39    2       0.11
       0.76    2       0.68
       3.00    2       0.22
       1.20    2       0.55

Multivariate test:
Test values   df   p-values
      10.03   10       0.44"""

        rt = ranktest(rng, f, 10000)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), rt) == """
Cointegration.TraceTest

 Rank    Value  p-value
    0   36.864    0.849
    1   21.054    0.865
    2   10.397    0.836
    3    4.067    0.703
    4    1.447    0.268"""

        f3 = setrank(f, 3)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), f3) == """
Cointegration.CivecmI1

α:
5×3 Matrix{Float64}:
 -0.0947422   0.0761861   0.0257547
  0.0632277   0.0336236   0.0637774
 -0.0428606  -0.0861965  -0.142879
  0.165899    0.146969   -0.065865
  0.168351   -0.118823    0.0502608

βᵀ:
3×5 Matrix{Float64}:
 0.150191    0.0384534  0.11206    0.0451019   -0.26955
 0.014997   -0.0768655  0.265054  -0.189269     0.268929
 0.0760236   0.145551   0.067286   0.00991004  -0.123557

Π:
5×5 Matrix{Float64}:
 -0.0111289  -0.0057506    0.0113096   -0.0184374    0.0428442
  0.0148491   0.00912971   0.0202887   -0.00288017  -0.0158808
 -0.0185921  -0.0158189   -0.0372635    0.0129653    0.00602609
  0.0221132  -0.0145042    0.0531135   -0.020987     0.00294423
  0.0273238   0.0229226   -0.00924725   0.0305804   -0.0835439"""
    end

    @testset "I(2) model" begin
        f = civecmI2(X, lags = 2)
        rt = ranktest(f)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), rt) == """
Cointegration.I2RankTest

5×6 Matrix{Float64}:
 768.781  591.41   430.957  280.984    149.057    36.8636
   0.0    428.372  274.61   139.985     24.4208   21.0544
   0.0      0.0    137.816   19.1122    13.2271   10.397
   0.0      0.0      0.0      9.40648    4.06859   4.06734
   0.0      0.0      0.0      0.0        1.44772   1.44657"""

        rng = StableRNG(123)
        rtsim = ranktest(rng, f, 100)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), rtsim[2]) == """
5×6 Matrix{Float64}:
 0.0  0.0  0.0  0.0   0.0   0.88
 0.0  0.0  0.0  0.0   0.91  0.87
 0.0  0.0  0.0  0.98  0.9   0.81
 0.0  0.0  0.0  0.94  0.98  0.75
 0.0  0.0  0.0  0.0   0.56  0.29"""

        f11 = setrank(f, 1, 1)
        @test sprint((io, t) -> show(io, MIME"text/plain"(), t), f11) == """
β':
1×5 Matrix{Float64}:
 0.00531565  0.0144423  0.0102059  -0.00801689  0.00657669

τ⊥δ:
5×1 Matrix{Float64}:
 -0.269794
 -0.46154
  0.743342
  0.344183
  0.497622

τ':
2×5 Matrix{Float64}:
 -0.153128  -0.385117   0.335879  -0.322867  -0.718646
 -0.22323   -0.615825  -0.618447   0.503593  -0.116684"""

        @testset "Convert to VAR and simulate" begin
            f_var = convert(VAR, f)
            f_sim = simulate(f_var, 10)
            @test f_sim isa Matrix
            @test size(f_sim) == (12, 5)
        end
    end
end

@testset "Illustrative examples from Johansen's book" begin
    danish_df = CSV.read(
        joinpath(dirname(pathof(Cointegration)), "..", "data", "danish.csv"),
        DataFrame,
    )
    danish_mat =
        [danish_df.log_real_money danish_df.log_real_income danish_df.bond_rate danish_df.deposit_rate]

    # Create constant term and seasonals.
    cnst = [ones(55) repeat(Matrix{Float64}(I, 4, 3), 14)[1:55, :]]
    # Johansen makes the seasonals orthogonal to the constant term
    proj(v, u) = (v'u) / (u'u) * u
    for k = 2:4
        cnst[:, k] .-= sum(proj(cnst[:, k], cnst[:, j]) for j = 1:k-1)
    end

    @testset "Chapter 2" begin
        f = civecmI1(danish_mat, unrestricted = cnst, lags = 2)

        @testset "Table 2.1" begin
            @test f.Π[:, 1:4] ≈ [
                -0.181 0.110 -1.042 0.638
                0.186 -0.309 0.658 -0.648
                0.014 -0.018 0.082 -0.167
                -0.004 0.020 0.143 -0.314
            ] atol = 2e-3
        end

        @testset "Table 2.2" begin
            @test f.Γ[:, 1:4] ≈ [
                0.195 -0.096 -0.138 -0.462
                0.504 -0.045 -0.377 0.060
                0.051 0.136 0.301 0.253
                0.069 -0.022 0.227 0.265
            ] atol = 2e-3
        end

        @testset "Table 2.3" begin
            @testset "The constant" begin
                @test f.Γ[:, 5] ≈ [1.583, -0.390, -0.064, -0.071] atol = 2e-3
            end

            @testset "The seasonal" begin
                @test_broken f.Γ[:, 6:8] ≈ [
                    -0.023 0.016 -0.039
                    -0.019 -0.007 -0.032
                    -0.003 -0.007 -0.007
                    -0.002 0.001 -0.003
                ]
            end
        end

        @testset "Table 2.4" begin
            @test cor(residuals(f)) ≈ [
                1.00 0.53 -0.45 -0.31
                0.53 1.00 -0.08 -0.24
                -0.45 -0.08 1.00 0.25
                -0.31 -0.24 0.25 1.00
            ] atol = 1e-2
        end

        @testset "Table 2.5" begin
            # FIXME! ARCH and JB tests not implemented yet
            @test mapslices(skewness, residuals(f), dims = 1) ≈ [0.552 0.524 -0.297 0.415] atol =
                1e-3
            @test mapslices(kurtosis, residuals(f), dims = 1) ≈ [-0.075 -0.087 0.576 0.562] atol =
                1e-3
        end

        @testset "Table 2.6" begin
            @test sort(eigvals(f), by = real, rev = true) ≈ [
                0.9725,
                0.7552 - 0.1571im,
                0.7552 + 0.1571im,
                0.6051,
                0.5955 - 0.3143im,
                0.5955 + 0.3143im,
                -0.1425 - 0.2312im,
                -0.1425 + 0.2312im,
            ] atol = 1e-3
        end
    end

    @testset "Chapter 7" begin
        # Create centered seasonal dummies
        cseason = repeat(Matrix{Float64}(I, 4, 3), 14)[1:55, :] .- 1 / 4
        f = civecmI1(danish_mat, exogenous = ones(55, 1), unrestricted = cseason)

        @testset "Table 7.1" begin
            rt = ranktest(f)
            # Notice that there is a typo in the book where it's 8.89 instead of 8.69 (correct value in the 1990 paper)
            @test rt.values ≈ [49.14, 19.06, 8.69, 2.35] atol = 1e-2
            @test all(rt.pvalues .> 0.05)
        end

        @testset "Table 7.2" begin
            # Signs flipped relative to book. Make first row positive
            # to make solution unique
            @test f.β .* sign.(f.β[1, :])' ≈ [
                21.9741 14.656 7.94655 1.02449
                -22.6981 -20.0509 -25.6408 -1.92976
                114.417 3.56115 4.27751 24.9971
                -92.6401 100.263 -44.8773 -14.6482
                -133.161 -62.5935 62.7489 -2.31866
            ] atol = 2e-2
        end

        @testset "Single coitegration relation" begin
            f1 = setrank(f, 1)
            @test f1.β / f1.β[1, 1] ≈ [1.00, -1.03, 5.21, -4.22, -6.06] atol = 2e-2
            @test f1.α * f1.β[1, 1] ≈ [-0.213, 0.115, 0.023, 0.029] atol = 2e-2

            @testset "restrictions on α and β" begin
                f1r1 = restrict(f1, Hβ = [1 0 0 0; -1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
                @test f1r1.β / f1r1.β[1, 1] * -21.55 ≈
                      [-21.55, 21.55, -114.22, 92.45, 134.99] atol = 2e-2
                # It appears that there are sign typos in the book and the magnitude is oof by 1000
                @test f1r1.α * f1r1.β[1, 1] / -21.55 * 1000 ≈ -[-9.83, 4.99, 1.05, 1.38] atol =
                    2e-2
                @test lrtest(f1r1, f1, 1).value ≈ 0.043 atol = 1e-2

                f1r2 = restrict(f1, Hβ = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1])
                @test lrtest(f1r2, f1r1, 1).value ≈ 0.89 atol = 1e-2

                f1r3 = restrict(f1r2, Hα = [1 0; 0 1; 0 0; 0 0])
                @test lrtest(f1r3, f1r2, 2).value ≈ 5.81 atol = 1e-2
            end
        end
    end
end

@testset "Juselius and Assenmacher (2017)" begin
    # Juselius, Katarina, and Katrin Assenmacher. "Real exchange rate persistence and the excess return puzzle: The case of Switzerland versus the US." Journal of Applied Econometrics 32, no. 6 (2017): 1145-1155.
    # https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.2562
    zipfile = ZipFile.Reader(download("http://qed.econ.queensu.ca/jae/datasets/juselius002/ja-data.zip"))
    csvfile = only(filter(f -> occursin("csv", f.name), zipfile.files))
    # It's quarterly data so eventually, we might want to build some functionality
    # around https://github.com/matthieugomez/PeriodicalDates.jl
    df = CSV.read(csvfile, DataFrame)
    close(zipfile)

    X = [df.LP df.LPus df.LShUS df.B10y df.USB10y df.R3M df.USR3m]
    # D = [month(df.Column1[i]) == j for i in 1:size(df, 1), j in 1:3]
    D_season = [month(df.Column1[i]) == j for i in 1:size(df, 1), j in 1:3]
    D_centered_season = D_season .- 1/4
    D_extra_season = [df.sea1D df.sea2D df.sea3D] # Should probably have been centered but they are not in the CATS file
    D_extra = [year(df.Column1[i]) == y && month(df.Column1[i]) == q for i in 1:size(df, 1), (y, q) in [(1980, 2), (1982, 4), (2008, 4)]]
    D_extra = [year(df.Column1[i]) == y && month(df.Column1[i]) == q for i in 1:size(df, 1), (y, q) in [(1980, 2), (1980, 4), (1981, 4), (1982, 4), (2008, 4)]]

    D = [D_centered_season D_extra_season D_extra]
    D = [ones(size(X, 1)) D_centered_season D_extra_season D_extra]
    D_nocentering = [D_season D_extra_season D_extra]
    D = [ones(size(X, 1)) D_season D_extra_season D_extra]
    Z = [float.(1:size(X, 1));;]

    ft = civecmI2(X; exogenous = Z, unrestricted = float.(D), lags = 2)
    ft = civecmI2(X[3:end,:]; exogenous = Z[3:end,:], unrestricted = float.(D[3:end,:]), lags = 2)

    rank_test_vals, rank_p_vals = ranktest(ft, 1000)
    rank_test_vals
    rank_p_vals

    ft_nocentering = civecmI2(X; exogenous = Z, unrestricted = float.(D_nocentering), lags = 2)
    ft_nocentering = civecmI2(X[3:end,:]; exogenous = Z[3:end,:], unrestricted = float.(D_nocentering[3:end,:]), lags = 2)

    rank_test_vals_nocentering, rank_p_vals_nocentering = ranktest(ft_nocentering, 1000)
    rank_test_vals_nocentering
    rank_p_vals_nocentering



    ft1 = civecmI1(X; exogenous = Z, unrestricted = D, lags = 2)
    ft1 = civecmI1(X; exogenous = Z, lags = 2)
    ranktest(ft1)

    ft2 = civecmI1(X - D*(D\X); exogenous = Z - D*(D\Z), lags = 2)
    ranktest(ft2)
end
