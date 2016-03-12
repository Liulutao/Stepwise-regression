function[bestModel, Cp]= stepwise(features, target, params)

% Stepwise regression includes regression models in which the choice of
% predictive variables is carried out by an automatic procedure.
% Usually, this takes the form of a sequence of F-tests.
% This realization in the capasity of program stop
% uses C_p Mallow's criterium.

% Arguments
% Input
%   features - matrix of features, where rows are objects, and colums are feature vectors
%   target   - target feature vector
%   params   - model parameters

% Output
%   bestModel- indices of informative features
%   Cp       - vector of Mallows's criterion depending on number of features          
X = features;
Y = target;
b= params;

[n,p]= size(X);
% table of Fisher
Ff= [161.45 18.51 10.13 7.71 6.61 5.99 5.59 5.32 5.12 4.96 4.84 4.75 4.67 4.60 4.54 4.49 4.45 4.41 4.38 4.35]';
Ff(21:1000)= 4.35;
A= zeros(n,p);  %matrix of add features
k=0;  %number of current features
Cp=zeros(n,1);  %Mallows criterion depending on number of features
C=[];   %Mallows criterion depending on step
SSE=[]; %SSE criterion depending on step
AIC=[];  %Akaike's criterion depending on step
MSE = sumsqr(Y - X*b)/n;
g=zeros(p, 1);  %indices of current features
G=[];  %matrix of indices
Bx= setdiff(1:p, 0);  %indices of remainig features
[F1 ]=Fisher(X, A, Y, b, Bx, k);  %find F criterion
[F, j]= max(abs(F1)) ; %find max F

while(F>= Ff(n-k))  %check the significance of feature
     k=k+1;  %increase number of features 
     A(:,j)= X(:,j);  %add new element
     g(j)= j;  %add index
     G= cat(1, G, g') ; %add index to matrix
     Cp(k)=sumsqr(Y - A*b)/MSE +2*k - n; %find Mallow's criterion
     C = cat(2, C, Cp(k));  
     SSEk= sumsqr(Y - A*b); %find SSE criterion
     SSE= cat(2, SSE, SSEk);
     AICk= 2*k+n*log(sumsqr(Y - A*b)/(n-2)); %find Akaike's criterion
     AIC= cat(2, AIC, AICk);
     Bx = setdiff(Bx,j);  %delete index j
     Ba = setdiff(1:p, Bx);
     [F1 ]=Fisher(X, A, Y, b, Bx, k);  %find F criterion
     [F,j]= max(abs(F1)); %find max F
end
[ F2]=Fisher(-A, A, Y, b, Ba, k);
[F, j] = min(abs(F2)); %find min F
while(F< Ff(n-k))  %check the significance of feature
     k=k-1;  %decrease number of features
     A(:,j)= 0;  %delete element
     g(j)= 0;  %del index
     G= cat(1, G, g') ;
     Cp(k) = sumsqr(Y - A*b)/MSE +2*k - n;  %find Mallow's criterion
     C= cat(2, C, Cp(k));
     SSEk= sumsqr(Y - A*b); %find SSE criterion
     SSE= cat(2, SSE, SSEk);
     AICk= 2*k+n*log(sumsqr(Y - A*b)/(n-2)); %find Akaike's criterion
     AIC= cat(2, AIC, AICk);
     Ba= setdiff(Ba,j);  %delete index j
     [ F2]=Fisher(-A, A, Y, b, Ba, k);  %find F criterion
     [F, j] = min(abs(F2));  %find min F
end
[minCp, idx] = min(C);% find the best model
bestModel = find(G(idx,:)); %find indices of informative features
%plot Cp, SSE, AIC criteria depending on step
[q, l]= size(C); 
i= 1:l;
hold on
plot(i, C, 'or-', 'LineWidth', 2)
plot(i, SSE, 'bs-', 'LineWidth', 2)
plot(i, AIC, 'mo-', 'LineWidth', 2)
xlabel('step');
ylabel('criteria');
legend('Cp', 'SSE', 'AIC')
text(2, C(2), '\leftarrow Stepwise')
hold off
end