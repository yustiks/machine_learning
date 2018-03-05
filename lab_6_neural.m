clear all;
%%%%%%%%% take 100 samples from each distribution
N = 100;
C1 = [2 1; 1 2];
C2 = [1 0; 0 1];
m1 = [0; 3];
m2 = [2; 1];
X1 = mvnrnd(m1,C1,N);
X2 = mvnrnd(m2,C2,N);
%%%%%%%%% plot the data points
figure(1),clf,
plot(X1(:,1),X1(:,2),'bx',X2(:,1),X2(:,2),'ro');
hold on;
grid on;
%%%%%%%%% add bias and create vector of classes
X11 = [X1,ones(N,1)];
X22 = [X2,ones(N,1)];
y = [-ones(N,1); ones(N,1)];
X = [X11; X22];
%%%%%%%% split our data into training and test sets
%ii = randperm(2*N);
%X_mix = X(ii,:);    %mixed Y
%y_mix = y(ii);      %mixed f
Xtr = X(1:160,:);
Xts = X(161:2*N,:);
ytr = y(1:160);
yts = y(161:2*N);

%%%%%%%% draw the quadratic boundary
MdlQuadratic = fitcdiscr(X,y,'DiscrimType','pseudoQuadratic');
MdlQuadratic.ClassNames([1 2])
K = MdlQuadratic.Coeffs(1,2).Const;
L = MdlQuadratic.Coeffs(1,2).Linear;
Q = MdlQuadratic.Coeffs(1,2).Quadratic;
f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h2 = ezplot(f,[-5 5 -5 5]);
h2.Color = 'r';
h2.LineWidth = 2;
%title('Bayes Decision Boundary')
xlabel('Feature X(1)')
ylabel('Feature X(2)')

%%%%%%%% draw the posterior probability as a 3D graph
x_vals = linspace(-4, 4, 50);
y_vals = linspace(-4, 4, 50);
p =zeros(length(x_vals));
for i = 1:length(x_vals)
    for j = 1:length(y_vals)
        x= [x_vals(i);y_vals(j)];
        gau_1 = 1/(det(C1)^.5)*exp(-.5*(x - m1)'*C1^-1*(x - m1));
        gau_2 = 1/(det(C2)^.5)*exp(-.5*(x - m2)'*C2^-1*(x - m2));
        p(i,j) = 1/(1+exp(log(gau_2/gau_1)));
    end
end
figure(2), clf
z = 0.5*ones(length(X1));
plot3(X1(:,1), X1(:,2), z, 'bx', X2(:,1), X2(:,2), z, 'ro');
hold on
surf(x_vals, y_vals, p')
%colorbar
axis([-4 4 -4 4 0 1])
title('Posterior probability of class 1 P[\omega_1|x]')
xlabel('Feature X(1)')
ylabel('Feature X(2)')
zlabel('Posterior probability of class 1 P[\omega_1|x]')

%%%%%%%% train a feedforward neural network
net = feedforwardnet(20);
net = train(net, X', y');
output = net(X');

%%%%%%%% evaluate the network output on the regular grid
figure(3),clf,
x = linspace(1,200,200);
scatter(x,output,'m*');

%%%%%%% plot the neural network decision boundary
%perf = perform(net, y', output);
len = 300;
x_vals = linspace(-10, 10, len);
y_vals = linspace(-10, 10, len);
vals = zeros(length(x_vals)^2,2);
for i = 1:len
    for j = 1:len
%       vals(i,j) = x_vals(i)*y_vals(j);
        vals(((i-1)*len+j),1) = x_vals(i);
        vals(((i-1)*len+j),2) = x_vals(j);
        vals(((i-1)*len+j),3) = 1;
    end
end
boundary = net(vals');
boundary = reshape(boundary, [len, len]);

z = 0*ones(length(X1));
figure(4), clf
plot3(X1(:,1), X1(:,2), z, 'bx', X2(:,1), X2(:,2), z, 'ro');
hold on;
surf(x_vals, y_vals, sign(boundary));
xlabel('Feature X(1)')
ylabel('Feature X(2)')
