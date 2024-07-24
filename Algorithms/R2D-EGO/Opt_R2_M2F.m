function new_x = Opt_R2_M2F(D,xlower,xupper,GPModels,ref_vecs,z,gmin,Batch_size)
% Maximizing N Subproblems and Selecting Batch of Query Points 
 
% This function was written by Liang Zhao.
 
    %% Use MOEA/D to maximize ETI
    [candidate_x,candidate_mean,candidate_std] = MOEAD_GR_(D,xlower,xupper,GPModels,ref_vecs,z,gmin);
    
    %% discard the duplicate candidates
    [candidate_x,ia,~] = unique(candidate_x,'rows'); 
    candidate_mean = candidate_mean(ia,:); candidate_std = candidate_std(ia,:);

    %% Compute EI_T for all the points in Q
	ETIs = zeros(size(candidate_x,1),size(ref_vecs,1));
    for j = 1 : size(candidate_x,1)
	    ETIs(j,:) = get_ETI(repmat(candidate_mean(j,:),size(ref_vecs,1),1),repmat(candidate_std(j,:),size(ref_vecs,1),1),ref_vecs,z,gmin);
    end

    %% find q solutions with the greedy algorithm
    Qb = subset_selection(ETIs,Batch_size);  
    new_x = candidate_x(Qb,:); 
end

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================
function  [pop_x,pop_mean,pop_std] = MOEAD_GR_(D,xlower,xupper,GPModels,ref_vecs,z,gmin) 
%% using MOEA/D-GR to solve subproblems in a collaborative manner
    maxIter  = 50; 
    pop_size = size(ref_vecs,1);

    %% neighbourhood   
    T       = ceil(pop_size/10); % size of neighbourhood
    B       = pdist2(ref_vecs,ref_vecs);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);

     %% the initial population for MOEA/D
    pop_x = (xupper-xlower).*lhsdesign(pop_size, D) + xlower; 
    [pop_mean,pop_std] = GPEvaluate(pop_x,GPModels);
    pop_ETI = get_ETI(pop_mean,pop_std,ref_vecs,z,gmin);   

    for gen = 1 : maxIter-1
       for i = 1 : pop_size
           if rand < 0.8
               P = B(i,randperm(size(B,2)));
           else
               P = randperm(pop_size);
           end
           %% generate an offspring and calculate its predictive mean and s
           off_x = operator_DE(pop_x(i,:),pop_x(P(1),:),pop_x(P(2),:), xlower,xupper);          
           [off_mean,off_std]= GPEvaluate(off_x,GPModels);  
            
           %% Global Replacement  MOEA/D-GR
           % Find the most approprite subproblem and its neighbourhood
           ETI_all = get_ETI(repmat(off_mean,pop_size,1),repmat(off_std,pop_size,1),ref_vecs,z,gmin);
           [~,best_index] = max(ETI_all);

           P = B(best_index,:);
           % Update the solutions in P
           offindex = P(pop_ETI(P)<ETI_all(P));
           if ~isempty(offindex)
               pop_x(offindex,:)    = repmat(off_x,length(offindex),1); % pop_x: N*D
               pop_mean(offindex,:) = repmat(off_mean,length(offindex),1);% pop_mean: N*M
               pop_std(offindex,:)  = repmat(off_std,length(offindex),1);% pop_std: N*M
               pop_ETI(offindex)    = ETI_all(offindex);% Pop_ETI: N*1
           end
       end      
    end
end

function Qb = subset_selection(ETIs,Batch_size)
    [L,N] = size(ETIs);
    Qb=[]; temp = ETIs;
    beta = zeros([1,N]); 
    for i = 1 : Batch_size
        [~,index] = max(sum(temp,2));
        Qb = [Qb,index];
        beta = beta + temp(index,:);
        % temp: [EI_T(x|w) - beta]_+
        temp = ETIs-repmat(beta,L,1);
        temp(temp < 0) = 0;   
    end
    Qb = unique(Qb); 
end
function  ETI = get_ETI(u,sigma,ref_vecs,z,Gbest)
%     g(x|w,z) = max{w1(f1-z1),w2(f2-z2)}  
% calculate the ETI(x|w) at multiple requests, e.g., N  
% u       : N*M  predictive mean
% sigma   : N*M  square root of the predictive variance
% ref_vecs: N*M  weight vectors 
% Gbest   : N*1  
% z       : 1*M  reference point   
    g_mu = ref_vecs.*(u - repmat(z,size(u,1),1));% N*M
    g_sig = ref_vecs.*sigma; % N*M
    % Moment Matching Approximation (MMA)
    g_sig(g_sig<0) = 0; g_sig2 = g_sig.^2; % N*M
     
     % Eq. 18 & Eq. 19 in MOEA/D-EGO
	[mma_mean,mma_sigma2] = app_max_of_2_Gaussian(g_mu(:,1:2),g_sig2(:,1:2)); % f1 & f2
    for i = 3 : size(g_mu,2)
        mu_temp = [mma_mean,g_mu(:,i)]; sig2_temp = [mma_sigma2,g_sig2(:,i)];
        [mma_mean,mma_sigma2] = app_max_of_2_Gaussian(mu_temp,sig2_temp);
    end
    
    mma_std = (sqrt(mma_sigma2));
    Gbest_minus_u = Gbest-mma_mean;
    tau = Gbest_minus_u./mma_std; % n*1

    % Precompute the normal distributions
    normcdf_tau = normcdf(tau);
    normpdf_tau = normpdf(tau);

    ETI = Gbest_minus_u.*normcdf_tau + mma_std.*normpdf_tau;
end

function [u,s] = GPEvaluate(X,model)
% Predict the GP posterior mean and std at a set of the candidate solutions 
    N = size(X,1); % number of samples
    M = length(model); % number of objectives
    u = zeros(N,M); % predictive mean
    MSE = zeros(N,M); % predictive MSE
    if N == 1 
        for j = 1 : M
            [u(:,j),~,MSE(:,j)] = Predictor(X,model{1,j}); % DACE Kriging toolbox
        end
        MSE(MSE<0) = 0;
    else
        for j = 1 : M
            [u(:,j),MSE(:,j)] = Predictor(X,model{1,j}); % DACE Kriging toolbox
        end
        MSE(MSE<0) = 0;
    end
   s = sqrt(MSE);% square root of the predictive variance
end

function [y,s2] = app_max_of_2_Gaussian(mu,sig2)
% Calculate  Eq. 18 & Eq. 19 in MOEA/D-EGO 
% n requests
% mu is N*2
% sig2 is N*2
    tao = sqrt(sum(sig2,2));  % N*1
    alpha = (mu(:,1)-mu(:,2))./tao;  % N*1
    % Eq. 16 / Eq. 18
    y = mu(:,1).*normcdf(alpha) + mu(:,2).*normcdf(-alpha) + tao.*normpdf(alpha);  % N*1
    % There is a typo in Eq. 17.  See Appendix B of MOEA/D-EGO.
    % It should be $$ +(\mu_1+\mu_2) \tau \varphi(\alpha)$$
    s2 = (mu(:,1).^2 + sig2(:,1)).*normcdf(alpha) + ...
        (mu(:,2).^2 + sig2(:,2)).*normcdf(-alpha) + (sum(mu,2)).*tao.*normpdf(alpha); 
%     s2 = (mu(:,1).^2 + sig2(:,1)).*normcdf(alpha) + ...
%         (mu(:,2).^2 + sig2(:,2)).*normcdf(-alpha) + (sum(mu,2)).*normpdf(alpha); 
    s2 = s2 - y.^2;
    s2(s2<0) = 0;
end

% >>>>>>>>>>>>>>>>    functions in PlatEMO ====================
function Offspring = operator_DE(Parent1,Parent2,Parent3, xlower,xupper)
%OperatorDE - The operator of differential evolution.
%
%   Off = OperatorDE(P1,P2,P3, xlower,xupper) uses the operator of differential
%   evolution to generate offsprings for problem Pro based on parents P1,
%   P2, and P3.  if P1, P2, and P3 are
%   matrices of decision variables, then Off is also a matrix of decision
%   variables, i.e., the offsprings are not evaluated. Each object or row
%   of P1, P2, and P3 is used to generate one offspring by P1 + 0.5*(P2-P3)
%   and polynomial mutation.
%
%------------------------------- Reference --------------------------------
% H. Li and Q. Zhang, Multiobjective optimization problems with complicated
% Pareto sets, MOEA/D and NSGA-II, IEEE Transactions on Evolutionary
% Computation, 2009, 13(2): 284-302.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [CR,F,proM,disM] = deal(1,0.5,1,20);
    [N,D] = size(Parent1);

    %% Differental evolution
    Site = rand(N,D) < CR;
    Offspring       = Parent1;
    Offspring(Site) = Offspring(Site) + F*(Parent2(Site)-Parent3(Site));

    %% Polynomial mutation
    Lower = repmat(xlower,N,1);
    Upper = repmat(xupper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end