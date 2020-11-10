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
scan = randi([1 8],1,1)

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
Y = sum(X_lsa,2);
%Y: 202 X 1 

%Mean center columns of (LSA) design matrix
nrow = size(X_lsa,1);
%nrow = 202, number of time points 
X_lsa = X_lsa - ones(nrow,1) * mean(X_lsa);
%X_lsa (202 X 24) - (202 X 1)* (1 x 24)

%Create LSS Design matrix
nStimTypes = 1;     
X_lss = NaN(nDown,nStimTypes+1,nTrials);
% X_lss : 202 X 2 (NULL+ MOVIE) X 24
for i = 1:nTrials
    currTrial = X_lsa(:,i);
    
    % Select all trials besides the trial of interest
    sel = ones(1,size(X_lsa,2));
    sel(1,i) = 0;
    sel = logical(sel);
    %only select current trial ()
    
    nuisPred = sum(X_lsa(:,sel),2);
    %202 x 1

    %Add predictors to design matrix for this trial
    X_lss(:,:,i) = [currTrial nuisPred];
    % 202 x 2 
end

%X_lss : 202 X 2 X 24 (Trials)


%add in noise for the Y
%from Mumford paper, but the actual sum(X_LSA,2) SD is 0.2 
Y_high = Y+normrnd(0,0.8,202,1);
Y_med = Y + normrnd(0,1.6,202,1);
Y_low = Y + normrnd(0,3,202,1);

%randomize sensored or spike index
window_list = [1:8];
LocSpike_list = [1:(length(Y)-1)];

%if the end position is less than zero(loc+window > length(Y)), than dont 

for w  = 1:length(window_list)
	nSpike = window_list(w);
	for l  = 1:length(LocSpike_list)

		if (LocSpike_list(l)+(window_list(w)-1)) <=  length(Y)
			spikeIdx = zeros(length(Y),1) ; 
			%SpikeIdx(startloc, startloc+windowsize-1) is where the censored data point is 
			spikeIdx(LocSpike_list(l):(LocSpike_list(l)+(window_list(w)-1))) = 1 ; 

			%Create spike regressors for design matrix
			spikeReg = zeros(nVols,nSpike);
			for spk = 1:nSpike
				spikeReg(LocSpike_list(l)+spk-1,spk) = 1; 
			end %spk
		end %if length correct 

		%spikeReg = [ones(length(spikeReg),1) spikeReg]; %commented out because right now doing everything in one matrix 
		%[intercept ; regress out motion point]

		%Zero out bad timepoints out of 202 trials from design matrix
        X_lss(logical(spikeIdx),:,:) = 0;


		%X_lss: timepoint(202)xstim/null x trials 
		nrow = size(X_lss,1);
		for dim = 1:size(X_lss,3)
		    X_lss(:,:,dim) = X_lss(:,:,dim) - ones(nrow,1) * mean(X_lss(:,:,dim));
		end 

		%see how to concatenate spike regression along axis 

		OutDesignMatrix.Motion = spikeReg;
		OutDesignMatrix.X = X_lss;
		%dimension 


		%[~,~,Y] = regress(Y,spikeReg);
		%[~,~,Y_high] = regress(Y_high,spikeReg);
		%[~,~,Y_med] = regress(Y_med,spikeReg);
		%[~,~,Y_low] = regress(Y_low,spikeReg);

		%OutBetaOrig{w,l} = extractBetaLSS(Y,X_lss);
		%OutBetaHigh{w,l} = extractBetaLSS(Y_high,X_lss);
		%OutBetaMed{w,l} = extractBetaLSS(Y_med,X_lss);
		%OutBetaLow{w,l} = extractBetaLSS(Y_low,X_lss);
		
		%Try to do it with concatenating along the X_lss
% 
% 		MotionReg = repmat(spikeReg,[1,1,nTrials]);
% 		X_lss_Motion = cat(2,X_lss,MotionReg);
% 
% 		OutBetaOrig_Xmotion{w,l} = extractBetaLSS(Y,X_lss_Motion);
% 		OutBetaHigh_Xmotion{w,l} = extractBetaLSS(Y_high,X_lss_Motion);
% 		OutBetaMed_Xmotion{w,l} = extractBetaLSS(Y_med,X_lss_Motion);
% 		OutBetaLow_Xmotion{w,l} = extractBetaLSS(Y_low,X_lss_Motion);
% 	
		
		%3 zero out Y but regress on regular regression model 
		Y_orig = sum(X_lsa,2);
		Y(logical(spikeIdx)) = 0;
		Y_high(logical(spikeIdx)) = 0;
		Y_med(logical(spikeIdx)) = 0;
		Y_low(logical(spikeIdx)) = 0;

		OutBetaOrig_X{w,l} = extractBetaLSS(sum(X_lsa,2),X_lss);
		OutBetaOrig0_X{w,l} = extractBetaLSS(Y,X_lss);
		OutBetaHigh_X{w,l} = extractBetaLSS(Y_high,X_lss);
		OutBetaMed_X{w,l} = extractBetaLSS(Y_med,X_lss);
		OutBetaLow_X{w,l} = extractBetaLSS(Y_low,X_lss);

	end %l 
end %w  

xplot = (1:24)';
figure
plot(xplot,OutBetaOrig_X{1,1}(1:8),'o')
hold on
plot(xplot,OutBetaLow_X{1,1:8}(1:8),'o')
plot(xplot,OutBetaHigh_X{1,1}(1:8),'o')
plot(xplot,OutBetaMed_X{1,1}(1:8),'o')
plot(xplot,OutBetaOrig0_X{1,1}(1:8),'o')