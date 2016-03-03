using Civecm

k = 5
n = 200

X = cumsum(randn(n, k))
f = civecmI1(X, 2)
ranktest(f)
setrank(f, 3)

f = civecmI2(X, 2)
ranktest(f)
setrank(f, 1, 1)
