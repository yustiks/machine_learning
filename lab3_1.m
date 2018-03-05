C = [2 1; 1 2];
X = randn(50,2);
Y = X*A;
N=50;
plotArray = zeros(N,1);
thRange = linspace(0,2*pi,N);
for n = 1:N
    t = thRange(n)
    u = [sin(t);cos(t)]
    !yp = Y*u;
    !var_empirical = var(yp)
    var_theoretical = u'*C*u
    plotArray(n) = var_theoretical
end
plot(plotArray)
    