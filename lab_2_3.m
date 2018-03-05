% Set up your own data in the following form
% X: N by 2 matrix of data
% y: class labels -1 or +1
% include column of ones for bias
X = [X ones(N,1)];

% Separate into training and test sets (check: >> doc randperm)
ii = randperm(N);
Xtr = X(ii(1:N/2),:); !first 50 
ytr = X(ii(1:N/2),:);
Xts = X(ii(N/2+1:N),:);!last 50
yts = X(ii(N/2+1:N),:);

% initialize weights
w = randn(3,1);

% Error correcting learning
eta = 0.001;
for iter=1:500
    j = ceil(rand*N/2);
    if ( ytr(j)*Xtr(j,:) < 0 )
        w = w + eta*ytr(j)*Xtr(j,:)';
    end
end

% Performance on test data
yhts = Xts*w;
disp([yts yhts])
PercentageError = 100*sum(find(yts .* yhts < 0))/Nts;
