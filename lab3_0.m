C = [2 1; 1 2];
theta = 0.25;
u = [sin(theta); cos(theta)]
A = chol(C);
X = randn(100000,2);
Y = X*A;
yp = Y*u;
var_empirical = var(yp)
var_theoretical = u'*C*u
