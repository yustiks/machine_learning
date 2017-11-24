C2 = [1.5 0; 0 1.5]
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);

quad_param = inv(C2) - inv(C1);
linear_param = 2 * (inv(C1) * m1 - inv(C2) * m2);
bias_param = m2' * inv(C2) * m2 - m1' * inv(C1) * m1 - log(sqrt(det(C1)) / sqrt(det(C2)));

[x3, y3] = meshgrid(-10:.2:10);

T_quad = x3.*x3.*quad_param(1,1) + 2*x3.*y3.*quad_param(1,2) + y3.*y3.*quad_param(2,2);
T_linear = x3.*linear_param(1) + y3.*linear_param(2);
T_bias = bias_param;

T = T_quad + T_linear + T_bias;
z3 = ones(101,101) ./ (1 + exp(-T));

hold off;
surf(x3,y3,z3);