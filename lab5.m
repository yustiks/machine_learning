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
Xtr = Y(1:450,:);
Xts = Y(451:N,:);
ytr = f(1:450,:);
fts = f(451:N,:);

%Set the widths of the basis functions to a sensibel scale 
%here the distance between two randomly chosen items of data

sig = norm(Xtr(ceil(rand*Ntr),:)-Xtr(ceil(rand*Ntr),:));

% Perform k-means clustering to find centres ck for the basis functions.
% Use K=Ntr/10 

K=Ntr/10
[Idx, C] = kmeans(Xtr, round(Ntr/10));

% Construct the design matrix

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


