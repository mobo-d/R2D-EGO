classdef R2DEGO < ALGORITHM
% <multi/many> <real/integer> <expensive>
% Multiobjective Efficient Global Optimization via R2-based Many-to-Few Decomposition
% batch_size --- 5 --- number of true function evaluations per iteration

%------------------------------- Reference --------------------------------
% L. Zhao, X. Huang, C. Qian, and Q. Zhang. Many-to-Few Decomposition: Linking
% R2-based and Decomposition-based Multiobjective Efficient Global Optimization 
% Algorithms. IEEE Transactions on Evolutionary Computation, 2024. 
%------------------------------- Acknowledge --------------------------------
% This implementation is based on PlatEMO. "Ye Tian, Ran Cheng, Xingyi Zhang,
% and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary multi-objective
% optimization [educational forum], IEEE Computational Intelligence Magazine, 
% 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function was written by Liang Zhao (liazhao5-c@my.cityu.edu.hk).
 
    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            batch_size = Algorithm.ParameterSet(5); 
            % number of initial samples
            n_init = 11*Problem.D-1;
            % Initial hyperparameters for GP
            theta = repmat({(n_init^(-1 ./ n_init)) .* ones(1, Problem.D)}, 1, Problem.M);
            
            %% Generate initial design using LHS
            x_lhs   = lhsdesign(n_init, Problem.D,'criterion','maximin','iterations',1000);
            x_init  = Problem.lower + (Problem.upper - Problem.lower).*x_lhs;  
            Archive = Problem.Evaluation(x_init);     
            
            %% Optimization
            while Algorithm.NotTerminated(Archive.best)
              %% Bulid GP model for each objective function  
                GPModels = cell(1,Problem.M);
                train_x = Archive.decs; train_y = Archive.objs;
                for i = 1 : Problem.M
                    GPModels{i}= Dacefit(train_x,train_y(:,i),'regpoly0','corrgauss',theta{i},1e-6*ones(1,Problem.D),20*ones(1,Problem.D));
                    theta{i}   = GPModels{i}.theta;
                end 
            
              %% Get weight vectors and Utopian point
                [ref_vecs,z,gmin] = get_weight_vectors(Problem.M,Problem.D,Problem.lower,Problem.upper,GPModels,train_y);

              %% Select multiple candidate points via R2-based M2F Decomposition
                Batch_size = min(Problem.maxFE - Problem.FE,batch_size); % the total budget is Problem.maxFE 
                new_x = Opt_R2_M2F(Problem.D,Problem.lower,Problem.upper,GPModels,ref_vecs,z,gmin,Batch_size);  
               
              %% Expensive Evaluation
                Archive = [Archive,Problem.Evaluation(new_x)];
            end
        end
    end
end