clear all;
clc;
folders = genpath('./Algorithms'); addpath(folders);
folders = genpath('./Metrics'); addpath(folders);
folders = genpath('./Problems'); addpath(folders);
  
n_run = 21; % # of runs
ins_list = {{'ZDT1',2,8,200},{'ZDT2',2,8,200},{'ZDT3',2,8,200},...
    {'ZDT4',2,8,200},{'ZDT6',2,8,200}, {'DTLZ1',3,6,300},...
    {'DTLZ2',3,6,300},{'DTLZ3',3,6,300},{'DTLZ4',3,6,300},...
    {'DTLZ5',3,6,300},{'DTLZ6',3,6,300},{'DTLZ7',3,6,300}};
 
for id = 1:1:length(ins_list)  
    clf;
    Problem = ins_list{id}; 
    prob_name = Problem{1,1}; 
    M = Problem{1,2}; D = Problem{1,3}; maxFE = Problem{1,4};
    score      = [];
    IGDps       = [];
    %% run DirHV-EGO
    for i = 1 : n_run
        Pro = feval(prob_name,'M',M,'D',D,'maxFE',maxFE); 
        Alg = R2DEGO('save',Inf,'run',i,'metName',{'IGDp'});% default batch size: 5
        Alg.Solve(Pro);
        % get metric value       
        IGDps = [IGDps;Alg.metric.IGDp(end)];
    end
    disp([sprintf('R2/D-EGO on %s_M%d_D%d_maxFE%d IGDp over %d runs:%.4e(%.2e)',prob_name,M,D,maxFE,n_run,mean(IGDps),std(IGDps))])
end
 
 
