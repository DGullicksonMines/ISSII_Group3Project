function sequenceLength = initializeSymbolMachine(filename)
% function sequenceLength = initializeSymbolMachine(filename)
%
% Initializes global variables used by the Symbol Machine. 
%
% Inputs:
%   filename: name of .mat file (such as 'dataset.mat', including quotes) 
%       containing one sequence of symbols stored in a vector called 
%       sequence. Each symbol in the sequence must be a digit from 1 to 9.
% 
% Outputs:
%   sequenceLength: the total number of symbols in the sequence
% 
% Created by Mike Wakin - 2023-03-07

clear global SYMBOLDATA
global SYMBOLDATA

eval(['load ' filename]);

sequenceLength = length(sequence);

SYMBOLDATA.filename = filename;
SYMBOLDATA.sequence = sequence;
SYMBOLDATA.sequenceLength = sequenceLength;
SYMBOLDATA.nextIndex = 1;
SYMBOLDATA.totalPenaltyInBits = 0;
SYMBOLDATA.correctPredictions = 0;
SYMBOLDATA.winnerProbabilities = zeros(sequenceLength,1);
SYMBOLDATA.loserProbabilities = zeros(sequenceLength,8);

