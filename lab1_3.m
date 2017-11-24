C = [2 1; 1 2];
A = chol(C)
B = A'*A;
X = randn(1000,2);
Y = X*A;
plot(X(:,1),X(:,2),'c.',Y(:,1),Y(:,2),'mx');