x = randn(1000,1);
!--hist(x,40);
[nn, xx] = hist(x,10);
bar(nn);