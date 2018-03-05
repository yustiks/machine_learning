X = X1
%compute mahalanobis mean distance accuracy 
euclidean1 = 0;
euclidean2 = 0;
mahalanobis1 = 0;
mahalanobis2 = 0;
eCorrect = 0;
mCorrect = 0;
for i = 1:N
    euclidean1 = (X(i,:)' - m1)' * (X(i,:)' - m1);
    euclidean2 = (X(i,:)' - m2)' * (X(i,:)' - m2);
    mahalanobis1 = (X(i,:)' - m1)'*inv(C1)*(X(i,:)'-m1);
    mahalanobis2 = (X(i,:)' - m2)'*inv(C2)*(X(i,:)'-m2);
    if euclidean1 <= euclidean2
        eCorrect = eCorrect + 1;    
    end
    if mahalanobis1 <= mahalanobis2
        mCorrect = mCorrect + 1;   
    end    
end
peCorrect = eCorrect * 100/N; 
e = num2str(peCorrect)
pmCorrect = mCorrect * 100/N; 
m = num2str(pmCorrect)

%nearest neighboar classifier
nCorrect = 0;
for jtst = 1:2*N
    xtst = X(jtst,:);
    ytst = y(jtst);
    jtr = setdiff(1:2*N,jtst);
    Xtr = X(jtr,:);
    ytr = y(jtr,1);
    for i = 1:2*N-1
        d(i) = norm(Xtr(i,:)-xtst);
    end
    [imin] = find(d==min(d));
    if(ytr(imin(1))*ytst>0)
        nCorrect = nCorrect + 1;
    %else
    %    disp('Incorrect classification');
    end
end
pCorrect = nCorrect * 100/(2*N);
nn = num2str(pCorrect)