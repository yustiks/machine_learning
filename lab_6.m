clear,
clc,
%
%Consider a two-class pattern classification problem
%in two dimensions, in which each class is Gaussian
%distributed with distinct means and covariance matrices
% m1 = (0 3)
% C1 = (2 1) (1 2)
% m2 = (2 1)
% C2 = (1 0) (0 1)

N = 100;
Xo = randn(N,2);
C1 = [2 1;1 2];
C2 = [1 0;0 1];

m1 = [0 3];
m2 = [2 1];

X1 = mvnrnd(m1,C1,100);
X2 = mvnrnd(m2,C2,100);
%
% Compute the posterior probability on a regular grid
%in the input space and plot the decision
%boundary for which the posterior probability
%satisfies
%P[w1|x] = 0.5
%
X1_no_bias = [X1',X2']';
bias = ones(200,1);
X1_bias = [X1_no_bias,bias];
Y = [ones(100,1)', -ones(100,1)']';
%%%%%%%% draw the quadratic boundary
figure,
MdlQuadratic = fitcdiscr(X1_bias,Y,'DiscrimType','pseudoQuadratic');
MdlQuadratic.ClassNames([1 2]);
K = MdlQuadratic.Coeffs(1,2).Const;
L = MdlQuadratic.Coeffs(1,2).Linear;
Q = MdlQuadratic.Coeffs(1,2).Quadratic;
f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h2 = ezplot(f,[-5 5 -5 5]);
h2.Color = 'r';
h2.LineWidth = 2;
title('The decision boundary for 2 classes');
hold on,

% 2 samples for each of the classes
%
%plot(X1(:,1), X1(:,2), 'bo', X2(:,1), X2(:,2),'rx');
%xlabel('x');
%ylabel('y');
%legend('C=[2 1][1 2] m=[0 3]', 'C=[1 0][0 1] m=[2 1]');
%title('2 samples drawn from each classes');
%figure,
%hold on;


% draw a posterior probability 3D graph
%
%-----------------this is not working!--------------------
%quad_param = inv(C2) - inv(C1);
%linear_param = 2 * (inv(C1) * m1' - inv(C2) * m2');
%bias_param = m2 * inv(C2) * m2' - m1 * inv(C1) * m1' - log(sqrt(det(C1)) / sqrt(det(C2)));
%[x3, y3] = meshgrid(-10:.2:10);

%T_quad = x3.*x3.*quad_param(1,1) + 2*x3.*y3.*quad_param(1,2) + y3.*y3.*quad_param(2,2);
%T_linear = x3.*linear_param(1) + y3.*linear_param(2);
%T_bias = bias_param;

%T = T_quad + T_linear + T_bias;
%z3 = ones(101,101) ./ (1 + exp(-T));

%hold on;
%surf(x3,y3,z3);
%xlabel('x');
%ylabel('y');
%title('Posterior probability as a three dimensional graph');
%
% ---------------that is working!-------------------------
%
%x_vals = linspace(-4, 4, 50);
%y_vals = linspace(-4, 4, 50);
%p =zeros(length(x_vals));
%for i = 1:length(x_vals)
%    for j = 1:length(y_vals)
%        x= [x_vals(i);y_vals(j)];
%        gau_1 = 1/(det(C1)^.5)*exp(-.5*(x - m1')'*C1^-1*(x - m1'));
%        gau_2 = 1/(det(C2)^.5)*exp(-.5*(x - m2')'*C2^-1*(x - m2'));
%        p(i,j) = 1/(1+exp(log(gau_2/gau_1)));
%    end
%end
%
%figure, clf
%z = 0.5*ones(length(X1));
%plot3(X1(:,1), X1(:,2), z, 'bx', X2(:,1), X2(:,2), z, 'ro');
%hold on
%surf(x_vals, y_vals, p')
%colorbar
%axis([-4 4 -4 4 0 1])
%title('Posterior probability of class 1 P[\omega_1|x]')
%xlabel('X')
%ylabel('Y')
%zlabel('Posterior probability of class 1 P[\omega_1|x]')
%figure,
% 
% 3D plot 
%

% Using the data sampled from each of the distributions, train a feedforward neural network
% using Matlab neural networks library 
% 
%[x,t] = simplefit_dataset;
%net = feedforwardnet(10);
%net = train(net,x,t);
%view(net)
%y = net(x);
%perf = perform(net,y,t)
%
% The following are commands you are likely to use
%
%1 hidden layer, a output layer
%
[net] = feedforwardnet(1); 
% 20 neurons to each layer
[net] = train(net, X1_bias', Y');

%view(net);
%After the network is trained and validated, 
%you can use the network object to calculate 
%the network response to any input.

k_train = 100;
X_train = ones(k_train,3);
Y_expected = ones(k_train,1);
r =randperm(200,k_train);
for i=1:k_train
    k = r(i);
    X_train(i,1) = X1_no_bias(k,1);
    X_train(i,2) = X1_no_bias(k,2);
    Y_expected(i) = Y(k);
end

Y_train = net(X_train');

k_class_1 = 0; 
for i=1:k_train
    if Y_train(i)==1
        k_class_1=1+k_class_1
    end
end
k_class_2 = k_train - k_class_1;
X_train_1 = ones(k_class_1,3);
X_train_2 = ones(k_class_2,3);
n1 = 1;
n2 = 1;
for i = 1:k_train
    if (Y_train(i)-1)^2<(Y_train(i)+1)^2
        X_train_1(n1,1)=X_train(i,1);
        X_train_1(n1,2)=X_train(i,2);
        n1 = n1+1;
    else
        X_train_2(n2,1)=X_train(i,1);
        X_train_2(n2,2)=X_train(i,2);
        n2 = n2+1;
    end
end
perfomance_neural1 = net(X_train_1');
perfomance_neural = net(X_train_2');
per_er = 0;
for i=1:n2-1
    per_er = per_er + (perfomance_neural(i)-(-1))^2;
end
for i=1:n1-1
    per_er = per_er + (perfomance_neural1(i)-1)^2;
end
per_er = per_er/((n2+n1-2)^2)
plot(X_train_1(:,1), X_train_1(:,2), 'go', X_train_2(:,1), X_train_2(:,2),'cx');


%{
hold on,
numGrid = 50,
xRange = linspace(-6.0, 4.0, numGrid);
yRange = linspace(-2.0, 7.0, numGrid);
xRange_net = linspace(-3.0, 4.0, numGrid);
yRange_net = linspace(-2.0, 7.0, numGrid);


p = zeros(50,50);
X_neuron = zeros(1,50);
Y_neuron = zeros(1,50); 
n = 50;
k_neuron = 1;
for i=1:n 
    for j=1:n
        p(i,j) = net([xRange_net(i),yRange_net(j),1]');
        if (p(i,j) <= 0.025 && p(i,j)>0)
            X_neuron(k_neuron) = xRange_net(i);
            Y_neuron(k_neuron) = yRange_net(j);
            k_neuron = k_neuron+1;
        end 
    end
end

X_show = X_neuron(1,1:k_neuron-1),
Y_show = Y_neuron(1,1:k_neuron-1),
plot(X_show, Y_show, 'bl');

%surface(xRange, yRange, p);

D = zeros(50,1);
for i=1:n
    D(i) = p(i,25)
end

        
title('The decision contour of neural network'),
perf = perform(net,Y_train,Y_expected')
%}
%
% 
%
%Pattern association showing error surface
%
%w_range = -1:0.1:1;
%b_range = -1:0.1:1;
%ES = errsurf(X,T,w_range,b_range,'purelin');
%plotes(w_range,b_range,ES);
%
%
%
%
%
%
