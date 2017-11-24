m1 = [0 2]';
m2 = [1.7 2.5]';
C1 = [2 1; 1 2];
C2 = [3 1.5; 1.5 3];

figure(11)
syms xf yf
a = [xf; yf];
boundary = a'*(inv(C1)-inv(C2))*a - 2*a'*inv(C1)*m1 - 2*a'*inv(C2)*m2
+ m1'*inv(C1)*m1 + m2'*inv(C2)*m2 + log(det(C1)/det(C2));
fimplicit(boundary,[-20 20 -20 20]);

figure(12)
[Xgrid,Ygrid] = meshgrid(-20:0.5:20,-20:0.5:20);
boundary = zeros(size(Xgrid));
for i=1:length(Xgrid(1,:))
    for j=1:1:length(Xgrid(:,1));
        a = [Xgrid(i,j); Ygrid(i,j)];
        boundary(i,j) = a'*(inv(C1)-inv(C2))*a - 2*a'*inv(C1)*m1 - 2*a'*inv(C2)*m2 + m1'*inv(C1)*m1 + m2'*inv(C2)*m2 + log(det(C1)/det(C2));
    end
end
Zbayes =(1 ./ (1 + exp((boundary))));
surf(Xgrid,Ygrid,Zbayes);