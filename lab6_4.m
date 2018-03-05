close all, clear all, clc, format compact
% data settings
N = 2000; % number of samples
Nu = 1500; % number of learning samples
% Mackay-Glass time series
b = 0.1;
c = 0.2;
tau = 17;
% initialization
y = [0.9697 0.9699 0.9794 1.0003 1.0319 1.0703 1.1076 1.1352 1.1485 ...
 1.1482 1.1383 1.1234 1.1072 1.0928 1.0820 1.0756 1.0739 1.0759]';
% generate Mackay-Glass time series
for n=18:N+99
 y(n+1) = y(n) - b*y(n) + c*y(n-tau)/(1+y(n-tau).^10);
end
% remove initial values
y(1:100) = [];
% plot training and validation data
plot(y,'m-')
grid on, hold on
plot(y(1:Nu),'b')
plot(y,'+k','markersize',2)
legend('validation data','training data','sampling markers','location','southwest')
xlabel('time (steps)')
ylabel('y')
ylim([-.5 1.5])
set(gcf,'position',[1 60 800 400])
% prepare training data
yt = con2seq(y(1:Nu)');
% prepare validation data
yv = con2seq(y(Nu+1:end)');
