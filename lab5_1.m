clear,
housing_data = load('housing.data');
[N, p1] = size(housing_data);
p = p1-1;
Y = [housing_data(:,1:p) ones(N,1)];
for j=1:p
    Y(:,j)=Y(:,j)-mean(Y(:,j));
    Y(:,j)=Y(:,j)/std(Y(:,j));
end
f = housing_data(:,p1);
f = f - mean(f);
f = f/std(f);

N = 506;

%Load the data, normalize it as done in Lab4 and get random positions of
%training and test set. 

Ntr = 450; 
Nts = N-Ntr;
Xtr = Y(1:450,:);
Xts = Y(451:N,:);
ytr = f(1:450,:);
yts = f(451:N,:);

%Set the widths of the basis functions to a sensibel scale 
%here the distance between two randomly chosen items of data

sig = norm(Xtr(ceil(rand*Ntr),:)-Xtr(ceil(rand*Ntr),:));

% Perform k-means clustering to find centres ck for the basis functions.
% Use K=Ntr/10 

K=Ntr/10
[Idx, C] = kmeans(Xtr, round(Ntr/10));


for i=1:Ntr
    for j=1:K
        A(i,j)=exp(-norm(Xtr(i,:)-C(j,:))/sig^2);
    end
end

% Solve for the weights
lambda = A\ytr; 

% Compute what the model predict at each of the training data: 

yh = zeros(Ntr, 1); 
u = zeros(K, 1); 
for n = 1:Ntr 
    for j = 1:K
        u(j) = exp(-norm(Xtr(n,:) - C(j,:))/sig^2);
    end
    yh(n) = lambda' * u;
end
plot(ytr, yh, 'rx', 'LineWidth', 2), grid on
title('RBF Prediction on Training Data', 'FontSize', 16);
xlabel('Target', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);

% Adapt the above to calculate what the model predicts at the unseen data
% (test data) and draw a similar scatter plot. How do the training and test
% errors compare? Compute the diffeence between training and test errors at
% different values of the number of basis functions, K. Briefly comment on
% any observation you make. 

sig_ts = norm(Xts(ceil(rand*Nts),:)-Xts(ceil(rand*Nts),:));

% Perform k-means clustering to find centres ck for the basis functions.
% Use K=Ntr/10 

[Idx_ts, C_ts] = kmeans(Xts, K);

% Construct the design matrix

for i=1:Nts
    for j=1:K
        A_ts(i,j)=exp(-norm(Xts(i,:)-C_ts(j,:))/sig_ts^2);
    end
end

% Solve for the weights
lambda_ts = A_ts\yts; 

% Compute what the model predict at each of the test data: 

yh_ts = zeros(Nts, 1); 
u_ts = zeros(K, 1); 
for n = 1:Nts 
    for j = 1:K
        u_ts(j) = exp(-norm(Xts(n,:) - C_ts(j,:))/sig_ts^2);
    end
    yh_ts(n) = lambda_ts' * u_ts;
end
figure
plot(yts, yh_ts, 'bx', 'LineWidth', 2), grid on
title('RBF Prediction on Test Data', 'FontSize', 16);
xlabel('Target', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);



etr_m = immse(yh, ytr);
ets_m = immse(yh_ts, yts);

% Calculate different for different K

% Perform k-means clustering to find centres ck for the basis functions.
%-----------------------------------------------------------
% Let`s see the different errors for different K : 
lr_ets_m = zeros(1,10);
rbf_etr_m = zeros(1,10);
rbf_ets_m = zeros(1,10);

a = linspace(10,50,10);
ii = randperm(N);
for idx = 1:numel(a)
    
    K=round(a(idx));
    [Idx, C] = kmeans(Xtr, K);

    A = zeros(Ntr, K);

    for i=1:Ntr
        for j=1:K
            A(i,j)=exp(-norm(Xtr(i,:)-C(j,:))/sig^2);
        end
    end

    % Solve for the weights
    lambda = A\ytr; 

    % Compute what the model predict at each of the training data: 

    yh = zeros(Ntr, 1); 
    u = zeros(K, 1); 
    for n = 1:Ntr 
        for j = 1:K
            u(j) = exp(-norm(Xtr(n,:) - C(j,:))/sig^2);
        end
        yh(n) = lambda' * u;
    end
    %plot(ytr, yh, 'rx', 'LineWidth', 2), grid on
    %title('RBF Prediction on Training Data', 'FontSize', 16);
    %xlabel('Target', 'FontSize', 14);
    %ylabel('Prediction', 'FontSize', 14);

    % Adapt the above to calculate what the model predicts at the unseen data
    % (test data) and draw a similar scatter plot. How do the training and test
    % errors compare? Compute the diffeence between training and test errors at
    % different values of the number of basis functions, K. Briefly comment on
    % any observation you make. 

    sig_ts = norm(Xts(ceil(rand*Nts),:)-Xts(ceil(rand*Nts),:));

    % Perform k-means clustering to find centres ck for the basis functions.

    [Idx_ts, C_ts] = kmeans(Xts, K);

    % Construct the design matrix
    A_ts = zeros(Nts, K);

    for i=1:Nts
        for j=1:K
            A_ts(i,j)=exp(-norm(Xts(i,:)-C_ts(j,:))/sig_ts^2);
        end
    end

    % Solve for the weights
%     lambda_ts = A_ts\yts; 

    % Compute what the model predict at each of the test data: 

    yh_ts = zeros(Nts, 1); 
    u_ts = zeros(K, 1); 
    for n = 1:Nts 
        for j = 1:K
            u_ts(j) = exp(-norm(Xts(n,:) - C_ts(j,:))/sig_ts^2);
        end
        yh_ts(n) = lambda' * u_ts;
    end

    rbf_etr_m(idx) = immse(yh, ytr);
    rbf_ets_m(idx) = immse(yh_ts, yts);
    
    
%     ans1(idx) = etr_m;
%     ans2(idx) = ets_m;
%     
    sample_size = 50;
    Y_mixed = Y(ii,:); %mixing the data 
    f_mixed = f(ii,:); %mixing the data

    Yts_10 = Y_mixed(1+(idx-1)*sample_size:idx*sample_size,:);
    fts_10 = f_mixed(1+(idx-1)*sample_size:idx*sample_size);
    
    [Ytr_90,indexes] = setdiff(Y_mixed,Yts_10,'rows');
    ftr_90 = f_mixed(indexes);
    
    w_vec = inv(Ytr_90'*Ytr_90)*Ytr_90'*ftr_90;
    lr_ets_m(idx) = sum((Yts_10*w_vec-fts_10).^2)/(sample_size)*100;
 

%    wtr = inv(Xtr'*Xtr)*Xtr'*ytr;
%    w = Xtr\ytr;
%    Ptr = Xtr*w;
%    Pts = Xts*w;
%    lr_etr_m(idx) = immse(Ptr, ytr);
%    lr_ets_m(idx) = immse(Pts, yts);

end

figure
plot(a',rbf_etr_m');
hold, 
plot(a',rbf_ets_m' ); 

title('The error of prediction RBF function', 'FontSize', 16);
xlabel('The value of K', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);

figure,
boxplot([lr_ets_m',rbf_ets_m'], {'Linear', 'RBF'}); title('The error of RBF and linear regression');





