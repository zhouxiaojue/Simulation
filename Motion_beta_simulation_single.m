BaseDir = pwd;
cd(BaseDir);
clear; clc
addpath(pwd)
addpath('/usr/local/MATLAB/R2017a/toolbox/fMRItoolbox/')
addpath('/home/xiaojue/bin/nifti_tools/')
addpath('/data1/2018_ActionDecoding/pilot/analysis_class/')
%addpath('/data1/2018_ActionDecoding/pilot/analysis_fc/')
addpath('/data1/2018_ActionDecoding/analysis_fc')

DataDir = '/data1/2018_ActionDecoding/data/';
OutFileDir = '/data1/2018_ActionDecoding/analysis_fc/';

SubPathPrefix = '/bv';
cd(DataDir)

task = 'instruction';
nVols = 202 ; %needs to check in the future, but for create design Data
%numRuns = 8;  %this was changed to NumVTCs for incomplete run 
nTrials = 24;

TR = 2;
sdmFlag = 1; % 1 = include 3DMC motion regressors
% OutideFiller = zeros(1, 1, 1, NumSeeds);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       make a list of subjects to analyze
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(DataDir); %enter the data directory
%change to list of pre-define subjects
%subList = dir('sub-*');


fileID = fopen ('/data1/2018_ActionDecoding/analysis_fc/InputList/Subjects_list.txt','r');
file = textscan(fileID,'%q');
subList = file{1};
fclose(fileID);

NumSubs = length(subList);

%DEBUG right now randomly choosing one subjects actual event file for simulation
sub =randi([1 length(subList)],1,1);

subID = char(subList(sub));
%example: sub-07
subPath = strcat(DataDir,subID, SubPathPrefix);
sdmPath = subPath;

behPath = strcat(DataDir, subID,'/beh/sess-02');
subPath_design = strcat(DataDir, subID);
OutPrefix  = subID;

% figure out how many vtcs that subject has
cd(subPath)
vtcList = dir('*actdecode*NATIVE.vtc');
NumVTCs = length(vtcList);


%2.Set up/Load the LSS design matrix
% set up some LSS parameters
%202x26x24 for each run volumes x number of regressions x total trails


cd(subPath_design)
designMotionfName = strcat(subID, '_LSSDespike.mat');

eventsList = dir([behPath '/*_events.tsv']);
behavList = dir([behPath '/*_responses.tsv']); 

%DEBUG right now dont need all scan event so randomly choosing one 
scan = randi([1 8],1,1);

baselineFlag = 0;

events = importdata([behPath '/' eventsList(scan).name]);
behav = importdata([behPath '/' behavList(scan).name]);

eventsTs = events.data(:,1:2:11);
eventsDur = eventsTs(:,2:end) - eventsTs(:,1:end-1);
eventsDur(:,6) = behav.data(:,1);   %add rt
eventsDur(:,7) = eventsDur(:,5) - eventsDur(:,6);   %Response interval - rt
eventsDur(:,5) = [];    %remove response interval dur (2.5 sec)
eventsDur(:,7) = [eventsTs(2:end,1) - eventsTs(1:end-1,6); 15];


events = eventsDur;

nTrials = size(events,1);   %number of movie clips presented
nEvents = size(events,2);
movieIdx = 3;               %Index of movie duration column in 'events'

%Construct vectors specifying durations of each event
times = reshape(permute(events,[2,1]),1,[]);    %vector of event durations
times = cumsum(times);                          %vector of event onsets relative to start
movieOnset = times(2:nEvents:end);
nullOnset = times(4:nEvents:end);
allCondsOnset = [movieOnset; nullOnset+2]; %%%%%% EG added time
allCondsOnset = allCondsOnset(:);

%Construct vector of stim lengths (for both movie and null portions)
stimLen = events(:,movieIdx);
stimLen = [stimLen repmat(3,length(nullOnset),1)]'; %'
stimLen = stimLen(:);

% Upsample Time resolution to .1 sec resolution
% See Mumford, Poldrack, Nichols (2011) for rationale
%      Add 147 seconds after last trial
TR = 1.5;
totTime = nVols*TR;
timeUp = 0:.1:totTime-1.5;
nUp = length(timeUp);
timeDown = timeUp(1:15:length(timeUp));
nDown = length(timeDown);

% Get model of hrf
hrf = twoGammaHrf( 30, .1, 0, 6, 16, ...
    1, 1, 6, 3 );
% Scale hrf so when convolved with really long boxcar
%   it will saturate at height of 1
hrf = hrf/sum(hrf);

%Create Design Matrix (LSA)
X_lsa = NaN(nDown,nTrials);

for i = 1:nTrials*2
   boxcar = (timeUp >= allCondsOnset(i)) .* (timeUp < (allCondsOnset(i)+stimLen(i)));
   pred = conv(hrf,boxcar);
   pred = pred(1:length(boxcar));
   pred = pred(1:15:end);
   X_lsa(:,i) = pred;
end


X_lsa = X_lsa(:,1:2:end);
Y = X_lsa(:,1);
Y = Y(find(Y ~= mode(Y)));
X = [ones(length(Y),1) Y];
%Y: 20 ish X 1 


%add in noise for the Y
%from Mumford paper, but the actual sum(X_LSA,2) SD is 0.2 
Y_high = Y + normrnd(0,0.8,length(Y),1);
Y_med = Y + normrnd(0,0.2,length(Y),1);
Y_low = Y + normrnd(0,0.1,length(Y),1);


Beta_Orig.beta = NaN(length(Y),length(Y)-1);
Beta_Orig.R =  NaN(length(Y),length(Y)-1);
Beta_Orig.betaErr =  NaN(length(Y),length(Y)-1);
Beta_Orig.RErr =  NaN(length(Y),length(Y)-1);

Beta_High.beta =  NaN(length(Y),length(Y)-1);
Beta_High.R = NaN(length(Y),length(Y)-1);
Beta_High.betaErr = NaN(length(Y),length(Y)-1);
Beta_High.RErr = NaN(length(Y),length(Y)-1);

Beta_Med.beta =  NaN(length(Y),length(Y)-1);
Beta_Med.R = NaN(length(Y),length(Y)-1);
Beta_Med.betaErr = NaN(length(Y),length(Y)-1);
Beta_Med.RErr = NaN(length(Y),length(Y)-1);

Beta_Low.beta =  NaN(length(Y),length(Y)-1);
Beta_Low.R =  NaN(length(Y),length(Y)-1);
Beta_Low.betaErr =  NaN(length(Y),length(Y)-1);
Beta_Low.RErr =  NaN(length(Y),length(Y)-1);

[tmpbeta,~,~,~,tmpR] = regress(Y,X);
Beta_NonCen.betaPerf = tmpbeta(2);
Beta_NonCen.RPerf = tmpR(1);
[tmpbeta,~,~,~,tmpR] = regress(Y_high,X);
Beta_NonCen.betaHigh = tmpbeta(2);
Beta_NonCen.RHigh = tmpR(1);
[tmpbeta,~,~,~,tmpR] = regress(Y_med,X);
Beta_NonCen.betaMed = tmpbeta(2);
Beta_NonCen.RMed = tmpR(1);
[tmpbeta,~,~,~,tmpR] = regress(Y_low,X);
Beta_NonCen.betaLow = tmpbeta(2);
Beta_NonCen.RLow = tmpR(1);


%randomize sensored or spike index
window_list = [1:length(Y)];
LocSpike_list = [1:(length(Y)-1)];

%if the end position is less than zero(loc+window > length(Y)), than dont 

for w  = 1:length(window_list)
	nSpike = window_list(w);
	for l  = 1:length(LocSpike_list)

		if (LocSpike_list(l)+(window_list(w)-1)) <=  length(Y)
			spikeIdx = zeros(length(Y),1) ; 
			%SpikeIdx(startloc, startloc+windowsize-1) is where the censored data point is 
			spikeIdx(LocSpike_list(l):(LocSpike_list(l)+(window_list(w)-1))) = 1 ; 

			Y_Orig_in = Y;
			Y_high_in = Y_high;
			Y_med_in = Y_med;
			Y_low_in = Y_low;

			Y_Orig_in(logical(spikeIdx)) = 0;			
			Y_high_in(logical(spikeIdx)) = 0;
			Y_med_in(logical(spikeIdx)) = 0;
			Y_low_in(logical(spikeIdx)) = 0;
 	 		

 	 		%R^2 first one 
			[tmpbeta,~,~,~,tmpR] = regress(Y_Orig_in,X);
			Beta_Orig.beta(w,l) = tmpbeta(2);
			Beta_Orig.R(w,l) = tmpR(1);
			Beta_Orig.betaErr(w,l) = Beta_NonCen.betaPerf - tmpbeta(2);  
			Beta_Orig.RErr(w,l) = Beta_NonCen.RPerf - tmpR(1);  

			[tmpbeta,~,~,~,tmpR] = regress(Y_high_in,X);
			Beta_High.beta(w,l) = tmpbeta(2);
			Beta_High.R(w,l) = tmpR(1);
			Beta_High.betaErr(w,l) = Beta_NonCen.betaHigh - tmpbeta(2);  
			Beta_High.RErr(w,l) = Beta_NonCen.RHigh - tmpR(1); 

			[tmpbeta,~,~,~,tmpR] = regress(Y_med_in,X);
			Beta_Med.beta(w,l) = tmpbeta(2);
			Beta_Med.R(w,l) = tmpR(1);
			Beta_Med.betaErr(w,l) = Beta_NonCen.betaMed - tmpbeta(2);  
			Beta_Med.RErr(w,l) = Beta_NonCen.RMed - tmpR(1); 

			[tmpbeta,~,~,~,tmpR] = regress(Y_low_in,X);
			Beta_Low.beta(w,l) = tmpbeta(2);
			Beta_Low.R(w,l) = tmpR(1);
			Beta_Low.betaErr(w,l) = Beta_NonCen.betaLow - tmpbeta(2);  
			Beta_Low.RErr(w,l) = Beta_NonCen.RLow - tmpR(1); 


        	end %if

	end %l 
end %w  



