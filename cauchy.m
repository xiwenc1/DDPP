%%
% code for the paper: Learning on Bandwidth Constrained Multi-Source Data with MIMO-inspired DPP MAP Inference
% Verification of Cauchy-Binet Formula 

%% initial value  


n=5;
m=10;

assert(n<m,'low rank! please make n<m')

x = rand(n,m);
y = rand(m,m);

%% Eq. (12)

sum_origianl = det(x*y*y'*x');


v=1:m;
c = nchoosek(v,n);

sum_1 = 0;

for i =1:length(c)
    sub = x*y;
   sum_1 = sum_1+ det(sub(:,c(i,:)))^2 ;
    
end



assert(abs(sum_1-sum_origianl)<1e-10,'They are not Equal')


%% Eq. (13)




sum_3 = 0;
for i =1:length(c)
    sub = x*y;
    sum2 = 0;
        for j =1:length(c)

          sum2 = sum2+ det( x(:,c(j,:)))*det( y(c(j,:),c(i,:)));


        end


    
    
    
   sum_3 = sum_3+ sum2^2 ;
    
end
 
assert(abs(sum_3-sum_origianl) <1e-10,'They are not Equal')

%% upperboud


up = 0;
for i =1:length(c)
    
    h =y*y';
   up = up+ det(h(c(i,:),c(i,:))) ;
    
end
up =up*det(x*x');


assert(up>sum_origianl,'Upperbound is less than origianl value')

sprintf('all good!')


















