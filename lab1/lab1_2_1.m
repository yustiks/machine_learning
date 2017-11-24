N = 1000;
x1 = zeros(N,1);
for n=1:N
    x1(n,1) = sum(rand(1000,1))-sum(rand(1000,1));
end
hist(x1,40);