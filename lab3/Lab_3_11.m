bayes_w = 2 * inv(C1) * (m2 - m1);
bayes_b = m1' * inv(C1) * m1 - m2' * inv(C2) * m2;

C2 = 1.5*C1
[x3, y3] = meshgrid(-10:.2:10);

z3 = ones(101,101) ./ (1 + exp(-(bayes_w(1) .* x3 + bayes_w(2) .* y3 + bayes_b)));
hold off;
surf(x3,y3,z3);
hold on;
plot3(X1(:,1),X1(:,2),.5*ones(200,1),'bx', X2(:,1), X2(:,2),.5*ones(200,1),'ro'); grid on;