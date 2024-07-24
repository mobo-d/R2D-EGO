function  [ref_vecs,z,gmin] = get_weight_vectors(M,D,xlower,xupper,GPModels,train_y,flag)
% generate and adjust the weight vectors

% This function was written by Liang Zhao.
% https://github.com/mobo-d/R2D-EGO

    % default：adjust the weight vectors
    if nargin == 6, flag = true; end
     
    %% Generate the initial weight vectors
    num_weights = [200,210,295,456,462]; % # of weight vectors：M = 2,3,4,5,6  
    if M <= 3
        [ref_vecs, ~]  = UniformPoint(num_weights(M-1),M); % simplex-lattice design 
    elseif M <= 6
        [ref_vecs, ~]  = UniformPoint(num_weights(M-1),M,'ILD'); % incremental lattice design
    else
        [ref_vecs, ~]  = UniformPoint(500,M); % Two-layered SLD
    end
  
    %% Estimate the Utopian point z
    [estimatedPF,z] = get_estimation_z(D,xlower,xupper,GPModels,ref_vecs,train_y); 

    %% adjust weight vectors
    if flag
        Zu = max(estimatedPF,[],1);
        temp = ref_vecs./repmat(Zu-z,size(ref_vecs,1),1);
        ref_vecs = temp./repmat(sum(temp,2),1, M); 
    end
    if  nargout > 2
        gmin = get_gmin(train_y,ref_vecs,z); 
    end
end

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================
function gmin = get_gmin(D_objs,ref_vecs,z)
% calculate the minimum of  Tch for each ref_vec
% g(x|w,z) = max{w1(f1-z1),w2(f2-z2),...}
    Objs_translated = D_objs-z; % n*M
    G = ref_vecs(:,1)*Objs_translated(:,1)';  % N*n, f1
    for j = 2:size(ref_vecs,2)
        G = max(G,ref_vecs(:,j)*Objs_translated(:,j)'); % N*n, max(fi,fj)
    end
    gmin = min(G,[],2);  % N*1  one for each weight vector 
end

function [estimatedPF,z] = get_estimation_z(D,xlower,xupper,GPModels,ref_vecs,train_y) 
% min(\mu_1(x),...,\mu_m(x))^T
% Utilize MOEA/D to minimize the GP posterior mean and determine the utopian
% point during optimization. Alternatively, other multi-objective optimization 
% algorithms such as NSGA-II can also be employed.

    delta=0.9; nr = 2; 
    maxIter = 100; 
    pop_size = size(ref_vecs,1);
    z = min(train_y,[],1);

    %% neighbourhood   
    T       = ceil(pop_size/10); % size of neighbourhood
    B       = pdist2(ref_vecs,ref_vecs);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);

    %% the initial population for MOEA/D
    pop_x = (xupper-xlower).*lhsdesign(pop_size, D) + xlower;
    pop_y = GPEvaluate_mean(pop_x,GPModels);
    z     = min(min(pop_y,[],1),z);
    for gen = 1 : maxIter-1
       for i = 1 : pop_size
           if rand < delta
               P = B(i,randperm(size(B,2)));
           else
               P = randperm(pop_size);
           end
           %% generate an offspring and calculate its predictive mean and s
           off_x = operator_DE(pop_x(i,:),pop_x(P(1),:),pop_x(P(2),:), xlower,xupper);    
           off_y = GPEvaluate_mean(off_x,GPModels);  
           z = min(z,off_y);        
           g_old = max((pop_y(P,:) - repmat(z,length(P),1)).*ref_vecs(P,:),[],2);
           g_new = max(repmat((off_y-z),length(P),1).*ref_vecs(P,:),[],2);
             
           % Update the solutions in P
           offindex = P(find(g_old>g_new,nr));
           if ~isempty(offindex)
               pop_x(offindex,:) = repmat(off_x,length(offindex),1); 
               pop_y(offindex,:) = repmat(off_y,length(offindex),1);
           end
       end      
    end
    [FrontNo,~] = NDSort(pop_y,1);
    estimatedPF = pop_y(FrontNo==1,:);
end

function [u] = GPEvaluate_mean(X,model)
% Predict the GP posterior mean at a set of the candidate solutions 
    N = size(X,1); % number of samples
    M = length(model); % number of objectives
    u = zeros(N,M); % predictive mean
    for j = 1 : M
        [u(:,j)] = Predictor(X,model{1,j}); % DACE Kriging toolbox
    end
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