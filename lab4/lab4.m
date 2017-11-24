BostonHousing = importdata('housing.data')
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