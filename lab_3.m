C1 = [2 1;1 2];
m1 = [0 2]';
m2 = [1.7 2.5]';

numGrid = 50; 
xRange = linspace(-6.0, 6.0, numGrid);
yRange = linspace(-6.0, 6.0, numGrid);
P1 = zeros(numGrid, numGrid);
P2 = P1;
for i=1:numGrid
    for j=1:numGrid
        x = [yRange(j) xRange(i)]';
        P1(i,j) = mvnpdf(x', m1', C1);
        P2(i,j) = mvnpdf(x', m2', C1);
    end
end
Pmax = max(max([P1 P2]));
!figure(1), clf,
!contour(xRange, yRange, P1, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
!hold on
!plot(m1(1), m1(2), 'b*', 'LineWidth', 4);
!contour(xRange, yRange, P2, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
!plot(m2(1), m2(2), 'r*', 'LineWidth', 4);

N = 200;
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C1, N);
!plot(X1(:,1),X1(:,2),'bx', X2(:,1),X2(:,2),'ro');
!grid on;
! Fisher
wF = inv(C1+C1)*(m1-m2);
xx = -6:0.1:6;
yy = xx*wF(2)/wF(1);
!plot(xx,yy,'r','LineWidth',2);

! Fisher discriminant projection
p1 = X1*wF;
p2 = X2*wF;
!random direction 
rd = [1-rand*2; 1-2*rand];
p1_rd = X1*rd;
p2_rd = X2*rd;
!projection onto the direction connecting the means of the two classes
means_vector = [1.7; 0.5]
p1_v = X1*means_vector; 
p2_v = X2*means_vector;

plo = min([p1; p2]);
phi = max([p1; p2]);
[nn1, xx1] = hist(p1);
[nn1_rd, xx1_rd] = hist(p1_rd);
[nn1_mv, xx1_mv] = hist(p1_v);
[nn2, xx2] = hist(p2);
[nn2_rd, xx2_rd] = hist(p2_rd);
[nn2_mv, xx2_mv] = hist(p2_v);

hhi = max([nn1 nn2 ]);
!figure(2),
!subplot(211), bar(xx1, nn1);
!axis([plo phi 0 hhi]);
!title('Destribution of Projections', 'FontSize', 16)
!ylabel('Class 1', 'FontSize', 14)
!subplot(212), bar(xx2, nn2);
!axis([plo phi 0 hhi])
!ylabel('Class 2', 'FontSize', 14)

! ROCCURVE
thmin = min([xx1 xx2  xx1_rd xx2_rd xx1_mv xx2_mv]);
thmax = max([xx1 xx2  xx1_rd xx2_rd xx1_mv xx2_mv]);

rocResolution = 50;
thRange = linspace(thmin, thmax, rocResolution);
ROC = zeros(rocResolution,2);
for jThreshold = 1:rocResolution
    threshold = thRange(jThreshold);
    tPos = length(find(p1 > threshold))*100/N;
    tPos_rd = length(find(p1_rd > threshold))*100/N;
    tPos_mv = length(find(p2_v > threshold))*100/N;
    fPos = length(find(p2 > threshold))*100/N;
    fPos_rd = length(find(p2_rd > threshold))*100/N;
    fPos_mv = length(find(p1_v > threshold))*100/N;
    ROC(jThreshold,:) = [fPos tPos];
    ROC_rd(jThreshold,:) = [fPos_rd tPos_rd];
    ROC_mv(jThreshold,:) = [fPos_mv tPos_mv];
    % accuracy 
    acs(jThreshold, :) = (length(find(p1 >= threshold)) + length(find(p2 < threshold))) * 50/200;
end

figure(3), clf,
plot(ROC(:,1), ROC(:,2), 'b', ROC_rd(:,1), ROC_rd(:,2), 'r', ROC_mv(:,1), ROC_mv(:,2), 'g');
!plot(ROC_rd(:,1), ROC_rd(:,2), 'b', 'LineWidth', 2);
!plot(ROC_mv(:,1), ROC_mv(:,2), 'b', 'LineWidth', 2);
axis([0 100 0 100]);
grid on, hold on
plot(0:100, 0:100, 'b-');
xlabel('False Positive', 'FontSize', 16);
ylabel('True Positive', 'FontSize', 16);
title('Receiver Operating Characteristic Curve', 'FontSize', 20);
! ROC Fisher
trapz(flipud(ROC(:,1)),flipud(ROC(:,2)))/10000
! ROC random
trapz(flipud(ROC_rd(:,1)),flipud(ROC_rd(:,2)))/10000
! ROC mean
trapz(flipud(ROC_mv(:,1)),flipud(ROC_mv(:,2)))/10000

!figure(4), clf,
!plot(thRange, acs)
!ylabel('Percentage of accuracy');