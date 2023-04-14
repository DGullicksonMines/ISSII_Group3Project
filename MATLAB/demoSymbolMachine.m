% Demonstration of Symbol Machine
% Created by Mike Wakin - 2023-03-07

% Initialize Symbol Machine with one of the provided sequences
sequenceLength = initializeSymbolMachine('sequence_nonuniform_train.mat');

% Stepping through the sequence one symbol at a time, your job is to 
% provide the Symbol Machine with a probabilistic forecast (a pmf) for the
% next symbol in the sequence. Since there are 9 possible symbols (the 
% digits 1 through 9), your pmf should have 9 entries that sum to 1. 

% One possible (but very simple) pmf that you could provide the Symbol 
% Machine would be to always forecase with the uniform pmf. Here is what 
% that would look like.
probs = ones(1,9)/9;
for ii = 1:sequenceLength
    [symbol,penalty] = symbolMachine(probs);
end

% After you have forecasted all of the entries in the sequence, the
% following function gives you a report of how good your predictions were.
reportSymbolMachine;

% As it turns out, the symbols in this particular sequence do not actually
% have a uniform pmf. Supposing we don't know in advance what the true pmf
% is, we can try to learn it along the way. In the following code, we start
% with a uniform pmf, but as we go, we reshape the pmf according to the
% symbols that we actually saw (up until now) in the sequence.
sequenceLength = initializeSymbolMachine('sequence_nonuniform_train.mat');
symbolCounts = ones(1,9); 
for ii = 1:sequenceLength
    probs = symbolCounts/sum(symbolCounts);
    [thisSymbol,penalty] = symbolMachine(probs);
    symbolCounts(thisSymbol) = symbolCounts(thisSymbol) + 1;
end
reportSymbolMachine;

