clc, clear,
a        = 0.2;     % value for a in eq (1)
b        = 0.1;     % value for b in eq (1)
tau      = 17;		% delay constant in eq (1)
x0       = 1.2;		% initial condition: x(t=0)=x0
deltat   = 1;	    % time step size (which coincides with the integration step)
sample_n = 2000;
% total no. of samples, excluding the given initial condition
interval = 1;	    % output is printed at every 'interval' time steps

time = 0;
index = 1;
history_length = floor(tau/deltat);
x_history = zeros(history_length, 1); % here we assume x(t)=0 for -tau <= t < 0
x_t = x0;

X = zeros(sample_n+1, 1); % vector of all generated x samples
T = zeros(sample_n+1, 1); % vector of time samples

for i = 1:sample_n+1,
    X(i) = x_t;
    if (mod(i-1, interval) == 0),
%         disp(sprintf('%4d %f', (i-1)/interval, x_t));
    end
    if tau == 0,
        x_t_minus_tau = 0.0;
    else
        x_t_minus_tau = x_history(index);
    end

    x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b);

    if (tau ~= 0),
        x_history(index) = x_t_plus_deltat;
        index = mod(index, history_length)+1;
    end
    time = time + deltat;
    T(i) = time;
    x_t = x_t_plus_deltat;
end


figure(1),
plot(T, X);
set(gca,'xlim',[0, T(end)]);
xlabel('t');
ylabel('x(t)');
title(sprintf('A Mackey-Glass time serie (tau=%d)', tau));

% 
%  Use the first N=1500 samples to train a prediction model 
% and the remaining 500 as test data. With p = 20, construct 
% the design matrix and output of a regression problem. 
% Your input matrix will have N-p+1 rows and p columns, 
% with each row being a time shifted version of the previous one
%
Xtrain = X(1:1500,:);
Ttrain = T(1:1500,:);
Xtest = X(1501:2000,:);
Ttest = T(1501:2000,:);
%
%
%design matrix 
%
%

matrix_num = 1481; 
ar_design = zeros(matrix_num, 20);
for i = 1:matrix_num,
    for j = 0:19 ,
        ar_design(i,j+1) = Xtrain(i+j);
    end
end
%
% then design matrix is used for prediction of next X
%   
% XB + e = y
% where B = ar_design
% f = y0
%
% calculation
% f - output
f = zeros(matrix_num,1);
for i = 1:matrix_num,
    f(i) = X(20+i);
end
%
% Estimate a linear predictor from the training data and check how well it does 
% one step ahead prediction on the test data
%

%linear regression 
% 

%
% linear regression plot 
%
% construct matrix for the 500 points
% 
matrix_tr = 480; 
ar_design_tr = zeros(matrix_tr, 20);
for i = 1:matrix_tr
    for j = 0:19 
        ar_design_tr(i,j+1) = Xtest(i+j);
    end
end

w_tr = (ar_design'*ar_design)\ar_design'*f;
output_linear = ar_design_tr*w_tr;
figure(1),
plot(Ttest(1:matrix_tr),output_linear,'b');
xlabel('time, T');
ylabel('$\frac{dx}{dt}=\frac{0.2x\left ( t-30  \right )}{1+x\left ( t-30\right )^{10}}-0.1x\left ( t \right )$','Interpreter','latex')

hold on, 
plot(Ttest(1:matrix_tr),X(2000-matrix_tr+1:2000),'rx');
legend('Mackey-Glass model','Linear prediction on time series');
error =(X(2000-matrix_tr+1:2000)-output_linear).^2;
figure(2),
%plot(Ttest(1:matrix_tr),error,'g');
bar(error);
xlabel('time, T');
ylabel('error = (ya-f)^2');
perfomance_error_linear = sum((X(2000-matrix_tr+1:2000)-output_linear).^2)/matrix_tr*100

% end linear regression
%
% free running mode 
% 


ar_linear = ar_design(481,20);
f_linear = f(500,1);
X_linear = T(1:1000,1);
Y_linear = X(1:500,1)';

for i=1:500
    y_linear = Y_linear(1,481+i-1:500+i-1);
    t = y_linear*w_tr;
    Y_linear = cat(2,Y_linear,t);
end

figure(3),
plot(X_linear, Y_linear', 'r');
hold on,
plot(T(1:1000,1), X(1:1000,1), 'b');
xlabel('time');
ylabel('x(t)');
title('Free running mode of linear regression');
legend('linear regression','Mackey-Glass time-series');

%
%
% Train a feedforward neural network and evaluate how well it performs on one step ahead 
% prediction
%

% ar_design_test - matrix for test 
[net] = feedforwardnet([10,10]);
[net] = train(net, ar_design', f');
X_network = X(1502:2000);
ar_design_test = zeros(499,20);
for i = 1:499
    temp = X(1481+i:1500+i)';
    ar_design_test(i,:) = temp(1,:);
end
output_network = net(ar_design_test');
figure(4),
plot(T(1482:1980), output_network,'b');
hold on, 
plot(T(1482:1980), X_network, 'r+');
legend('neural network performance','time-series output data');
xlabel('time,T');
ylabel('$\frac{dx}{dt}=\frac{0.2x\left ( t-30  \right )}{1+x\left ( t-30\right )^{10}}-0.1x\left ( t \right )$','Interpreter','latex');
%

error1 = (X_network-output_network').^2;
figure(5),
plot(T(1502:2000),error1,'g');
xlabel('time, T');
ylabel('error = (f1-f)^2');
perfomance_neural = perform(net,X_network,output_network')
%
% free running mode for neural network 
% 
%
%[net] = feedforwardnet(20); 
% 20 neurons to each layer
%ii = randperm(1481);
%ar_design_mix = ar_design(ii, :); % mixing the data
%f_net_mixed = f(ii, :); % mixing the data
%ar_design_net = ar_design_mix(1:100,:);
%f_net = f_net_mixed(1:100,:);

%[net] = train(net, ar_design_net', f_net');
%
%Y_expected = net(ar_design');
%perfomance_neural = perform(net,f,Y_expected')
%
% Use the trained neural network in a free running mode, feeding back predicted outputs 
% feeding back into the input and check if sustained oscillations are possible
X_net = T(1:400,1);
Y_net = X(1:200,1)';
for i=1:200
    t = Y_net(1,200-20+i:200+i-1);
    y_net = net(t');
    Y_net = cat(2,Y_net,y_net);
end
figure(6),
plot(X_net,Y_net,'g');
hold on,
plot(X_net,X(1:400),'r');
hold on,
y_plot = linspace(-4,5,50);
x_plot = zeros(50,1);
for i=1:50
    x_plot(i)=200;
end
plot(x_plot, y_plot,'b.');
xlabel('time');
ylabel('x(time)');
title('Feedforward neural network prediction in a free running mode');
legend('neural network prediction','Mackey-Glass time-series');

function x_dot = mackeyglass_eq(x_t, x_t_minus_tau, a, b)
    x_dot = -b*x_t + a*x_t_minus_tau/(1 + x_t_minus_tau^10.0);
end

function x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b)
    k1 = deltat*mackeyglass_eq(x_t,          x_t_minus_tau, a, b);
    k2 = deltat*mackeyglass_eq(x_t+0.5*k1,   x_t_minus_tau, a, b);
    k3 = deltat*mackeyglass_eq(x_t+0.5*k2,   x_t_minus_tau, a, b);
    k4 = deltat*mackeyglass_eq(x_t+k3,       x_t_minus_tau, a, b);
    x_t_plus_deltat = (x_t + k1/6 + k2/3 + k3/3 + k4/6);
end
