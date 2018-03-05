%m1 = [0 2]';
%m2 = [1.7 2.5]';
%X = [X1; X2];
%distance to mean classifier
cor1 = 0;
cor2 = 0;
for i = 1:(200)
    %distance from each point to mean1 and mean2
    x1_mean1 = norm(X1(i,:)'-m1);
    x1_mean2 = norm(X1(i,:)'-m2);
    maha_x1_mean1 = (X1(i,:)'-m1)'*C1^(-1)*(X1(i,:)'-m1);
    maha_x1_mean2 = (X1(i,:)'-m2)'*C1^(-1)*(X1(i,:)'-m2);
    if x1_mean1<=x1_mean2
        cor1 = cor1+1;
    end
    if maha_x1_mean1 <=maha_x1_mean2
        cor2 = cor2+1;
    end
    x2_mean1 = norm(X2(i,:)'-m1);
    x2_mean2 = norm(X2(i,:)'-m2);
    maha_x2_mean1 = (X2(i,:)'-m1)'*C1^(-1)*(X2(i,:)'-m1);
    maha_x2_mean2 = (X2(i,:)'-m2)'*C1^(-1)*(X2(i,:)'-m2);
    if x2_mean2<=x2_mean1
        cor1 = cor1+1;
    end
    if maha_x2_mean2<=maha_x2_mean1
        cor2 = cor2+1;
    end
end
pCorrect1 = cor1/4;
pCorrect2 = cor2/4;
disp(['E: ' num2str(pCorrect1)]);
disp(['Mahalanobis distance: ' num2str(pCorrect2)]);