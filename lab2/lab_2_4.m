N = 100;
Xo = randn(N,2);
C = [2 1; 1 2];
A = chol(C);
x = Xo * A;
!for n=1:100
n=40
    N=100;
    m1 = [0 n*0.25];
    m2 = [n*0.25 0];
    X1 = x + kron(ones(N,1), m1);
    X2 = x + kron(ones(N,1), m2);
    X11 = [X1,ones(N,1)];
    X22 = [X2,ones(N,1)];
    y = [-ones(N,1); ones(N,1)];
    X = [X11; X22];

    % Separate into training and test sets (check: >> doc randperm)
    N = 200;
    ii = randperm(N);
    Xtr = X(ii(1:N/2),:);
    ytr = y(ii(1:N/2),:);
    Xts = X(ii(N/2+1:N),:);
    yts = y(ii(N/2+1:N),:);


    % initialize weights
    w = randn(3,1);

    %calculate our expected y
    %how do I know what is the desired output?
    %any point from X1 we expect to be class 1, any point from X2 class -1 

    % Error correcting learning
    eta = 0.1;
    for iter=1:10000
        j = ceil(rand*N/2);
        if ( ytr(j)*Xtr(j,:)*w < 0 )
            w = w + eta*ytr(j)*Xtr(j,:)';
        end
    end

    % Performance on test data
    yhts = Xts*w;
    %disp([yts yhts]);
    PercentageError = sum(yts .* yhts < 0)
    x111(n)=n;
    y111(n)=PercentageError;
!end
!plot(x111,y111);
plot(X1(:,1),X1(:,2),'b.',X2(:,1),X2(:,2),'r.');
hold on;
x_plot = [linspace(-5,5,100)];
plot(x_plot,-(w(1)*x_plot)/w(2));

!plot(X2(:,1),X2(:,2),'c.',Y2(:,1),Y2(:,2),'mx');
!hold on;
C1=C^(-1)
w = 2*C1*(m2-m1)';
b = m1*C1*m1' - m2*C1*m2';
a1 = w(1);
b1 = w(2);
c1 = b
x1 = -4
y1 = (-c1-a1*x1)/b1
x2 = 6
y2 = (-c1-a1*x2)/b1
title('Img7 Perceptron algorithm with m1=[0 10] m2=[10 0]')
plot([x1 x2],[y1 y2],'g');
!legend('class1','class2','perceptron','the Bayes’ optimal boundary','Location','northwest')