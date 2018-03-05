N = 100;
C = [2 1; 1 2]
A = chol(C)
X = randn(100,2);
m1 = [0 2];
m2 = [1.5 0];
X1 = X + kron(ones(N,1), m1);
X2 = X1*A;
Y1 = X + kron(ones(N,1), m2);
Y2 = Y1*A;
C1 = C^(-1);
w = 2*C1*(m2-m1)';
b = m1*C1*m1' - m2*C1*m2';
a1 = w(1);
b1 = w(2);
c1 = b
x1 = -4
y1 = (-c1-a1*x1)/b1
x2 = 6
y2 = (-c1-a1*x2)/b1
plot(X2(:,1),X2(:,2),'c.',Y2(:,1),Y2(:,2),'mx');
hold on;
plot([x1 x2],[y1 y2],'g');

