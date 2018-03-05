%filename = 'text_finance.mat'
%textdata = textdata(2:1266,1);
%save(filename, '-mat', 'textdata')
clear,
clc,
% load data
filename = 'finance.mat';
%save(filename,'data'),
load(filename,'-mat','data');
filename = 'text_finance.mat';
load(filename, '-mat', 'textdata');
%
%
% Obtain daily FTSE100 data for the past five years from a financial data provider e.g. finance.yahoo.com
% Formulate a neural network predictor that predicts tomorrows FTSE index value from 20 past trading days.
% Use market Close prices. Was there an opportunity to make money using your knowledge of neural networks? 
% Does the ability to predict stock index improve if you were to use past values of Volume Traded information
% as additional input? 
%
nn = 1265;
T = zeros(1,nn);
volume = zeros(1,nn);
f = zeros(1,nn);
for i=1:nn
    T(i) = i;
    f(i) = data(i,2);
    volume(i) = data(i,5);
end


%
startDate = datenum('12-03-2012');
% Select an ending date.
endDate = datenum('12-01-2017');
% Create a variable, xdata, that corresponds to the number of years between the start and end dates.
%xData = linspace(startDate,endDate,nn);
%figure(7),
%plot(xData,f,'b');
%datetick('x','yyyy','keeplimits');
%
%
% create a design matrix 
% 
ntr = 1200;
ad = zeros(ntr-20,20);
f_out = zeros(ntr-20,1);
for i = 1:ntr-20+1
    ad(i,:) = f(1,i:i+19)';
    f_out(i,:) = f(1,i+20);
end
%
[net] = feedforwardnet([10,10]); 

[net] = train(net, ad', f_out');
%
% design matrix for the left data
%

matrix_design_finance = zeros(1245-1181,20);
t1 = ad(1181,2:20);
t2 = f_out(1181,1);
matrix_design_finance(1,:) = cat(2, t1, t2);
for i=1183:1246
    % put the result of again to the output
    t2 = matrix_design_finance(i-1182,:);
    out_finance = net(t2');
    t1 = matrix_design_finance(i-1182,2:20);
    matrix_design_finance(i-1181,:) = cat(2, t1, out_finance); 
end
figure(8),
%

startDate = datenum('09-4-2017');
% Select an ending date.
endDate = datenum('12-01-2017');
% Create a variable, xdata, that corresponds to the number of years between the start and end dates.
xData = linspace(startDate,endDate,65);
%}

plot(xData',matrix_design_finance(:,20),'r');
hold on,
f_real_finance = f(1,1201:1265)';
plot(xData',f_real_finance,'b');
legend('predicted data','real data'),
set(gca,'XTick',xData),
datetick('x','mm/dd','keepticks'),

%
% let`s make just prediction from the 20 days
%
mtr_finance = zeros(65,20);
f_finance_ = zeros(65,1);
for i = 1:65
    temp_f = f(1, 1180+i:1199+i);
    mtr_finance(i,:) = temp_f(1,:); 
    f_finance_(i) = net(mtr_finance(i,:)');
end
figure(9),
plot(xData', f_finance_,'r');
hold on,
f_print = f(1,1201:1265);
plot(xData', f_print','b');
legend('predicted data','real volume data');
set(gca,'XTick',xData),
datetick('x','mm/dd/yy');

errors = zeros(1,65);
for i = 1:65
    errors(1,i) = (f_print(1,i)-f_finance_(i,1)').^2;
end 
figure(10),
%plot(xData', errors','b');
bar(errors);
xlabel('time, T');
ylabel('squared errors');
set(gca,'XTick',xData),
datetick('x','mm/dd/yy');

% neural network depends on 2 vectors
%
%
% last part
%
%
f1 = f;
volume1 = volume;
f_=zeros(1,1180);

f1(1,:)=f1(1,:)-mean(f1(1,:));
f1(1,:)=f1(1,:)/std(f1(1,:));
volume1(1,:)=volume1(1,:)-mean(volume1(1,:));
volume1(1,:)=volume1(1,:)/std(volume1(1,:));


ar_design_ = zeros(1180,40);
for i=1:1180
    ar_design_(i,1:20) = f1(1,i:i+19);
    ar_design_(i,21:40) = volume1(1,i:i+19);
    f_(i)=f1(1,i+20);
end

[net1] = feedforwardnet(20);
[net1] = train(net1, ar_design_', f_);

% create design matrix for test set 
results_ = zeros(65,1);

ar_design_test_ = zeros(65,1);
for i=1:65
    ar_design_test_(i,1:20) = f1(1,1180+i:1180+i+19);
    ar_design_test_(i,21:40) = volume1(1,1180+i:1180+i+19);
    results_(i) = net1(ar_design_test_(i,:)');
end


figure(12),
plot(xData,results_,'b');
hold on,
results_real=f1(1,1201:1265); 
plot(xData,results_real,'g');
set(gca,'XTick',xData),
datetick('x','mm/dd/yy');
xlabel('time, T');
ylabel('close price and volume price normalised data');
legend('predicted data','real data');

errors = zeros(1,65);
for i = 1:65
    errors(1,i) = (results_real(1,i)-f_finance_(i,1)').^2;
end 
figure(13),
%plot(xData', errors','b');
bar(errors);
xlabel('time, T');
ylabel('squared errors');
set(gca,'XTick',xData),
datetick('x','mm/dd/yy');

w = net1.IW{1};
ar = zeros(1,40);
for j = 1:40
    sum = 0
    for i = 1:20
        sum = sum + w(i,j)*w(i,j);
    end
    ar(1,j)=sum/(40*40);
end

figure(14),
bar(ar);
xlabel('40 weights of two data set of 20 days: 20 weigths for close price, 20 days for volume price')
ylabel('squared mean weights');
