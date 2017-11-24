BostonHousing = importdata('housing.data')
corrcoef(BostonHousing);
[N, p1] = size(BostonHousing);
p = p1-1;
Y = [BostonHousing(:,1:p) ones(N,1)];
for j=1:p
    Y(:,j)=Y(:,j)-mean(Y(:,j));
    Y(:,j)=Y(:,j)/std(Y(:,j));
end
f = BostonHousing(:,p1);
f = f - mean(f);
f = f/std(f);

w = inv(Y'*Y)*Y'*f;
fh = Y*w;
figure(1), clf, 
plot(f, fh, 'r.', 'LineWidth', 2), 
grid on
xlabel('True House Price', 'FontSize', 14)
ylabel('Prediction', 'FontSize', 14)
title('Linear Regression', 'FontSize', 14) 

%Split the data into a training set and a test set
%estimate the regression model (w) on the training set 
%and see how training set and see how training and test errors differ
ytr = Y(1:400,:);
yts = Y(401:N,:);
ftr = f(1:400,:);
fts = f(401:N,:);

wtr = inv(ytr'*ytr)*ytr'*ftr;
etr_m = mean((ytr*wtr-ftr).^2)
ets_m = mean((yts*wtr-fts).^2)
hold on
%plot([-2,3],[-2,3],'b') 

%10-fold cross validation
nn = 10;
error = zeros(nn,1);
sample_size = 50;

ii = randperm(N);
Y_mixed = Y(ii, :); % mixing the data
f_mixed = f(ii, :); % mixing the data
for i = 1:nn
    Yts_10 = Y_mixed(1+(i-1)*sample_size:i*sample_size,:);
    fts_10 = f_mixed(1+(i-1)*sample_size:i*sample_size);
    
    [Ytr_90,indexes] = setdiff(Y_mixed,Yts_10,'rows'); % all data without 50 points
    ftr_90 = f_mixed(indexes);
    
    w_vec = inv(Ytr_90'*Ytr_90)*Ytr_90'*ftr_90;
    error(i) = sum((Yts_10*w_vec-fts_10).^2)/(sample_size)*100;
end

figure
hist(error);
xlabel('Value of error', 'FontSize', 14)
ylabel('Number of errors', 'FontSize', 14)
title('Histogram of errors in 10-step cross validation', 'FontSize', 14) 
average_error = mean(error)
uncertainty = std(error)

%Regression using the CVX Tool:
cvx_begin quiet
variable w1( p+1 );
minimize norm( Y*w1 - f )
cvx_end
fh1 = Y*w1;
%Check if the two methods produce the same results.
figure(3), clf,
plot(w, w1, 'mx', 'LineWidth', 2);
%Let us now regularize the regression: w2 = minw jY w ? fj + jwj1. 
%You can implement
%this as follows:
gama = 8.0;
cvx_begin quiet
variable w2( p+1 );
minimize( norm(Y*w2-f) + gama*norm(w2,1) );
cvx_end
fh2 = Y*w2;
plot(f, fh1, 'co', 'LineWidth', 2),
hold on,
plot(f, fh2, 'r+', 'LineWidth', 2),
legend('Regression', 'Sparse Regression');
xlabel('True House Price', 'FontSize', 14)
ylabel('Prediction', 'FontSize', 14)
title('Different methods of linear regression', 'FontSize', 14) 
%You can find the non-zero coefficients that are not switched off by the regularizer:
[iNzero] = find(abs(w2) > 1e-5);
disp('Relevant variables');
disp(iNzero);

gammas = (linspace(0.01, 40, 100))';
k = zeros(100, 1);

for g=1:100
    gamma = gammas(g);
    
    cvx_begin quiet
    variable w3( p+1 );
    minimize( norm(Y*w3-f) + gamma*norm(w3,1) );
    cvx_end
    fh2 = Y*w3;

    %You can find the non-zero coefficients that are not switched off by the regularizer:
    [nZero] = find(abs(w3) > 1e-5);
    k(g) = size(nZero, 1);
    disp(nZero);
end

figure(4), clf,
plot(gammas, k);
ylabel('Number of non-zero coefficients', 'FontSize', 14)
xlabel('Value of gamma', 'FontSize', 14) 
title('The number of non-zero coefficients', 'FontSize', 14) 
