function reportSymbolMachine
% function reportSymbolMachine
%
% Creates a report on the quality of the forecasts fed into the Symbol
% Machine. Should be used after the whole sequence has been parsed by
% the Symbol Machine, i.e., after symbolMachine.m has been called once for
% each symbol appearing in the sequence.
%
% Inputs/outputs: none.
%
% Created by Mike Wakin - 2023-03-07

global SYMBOLDATA

fprintf('\n');
fprintf('Symbol Machine processed %d out of %d symbols in %s.\n',SYMBOLDATA.nextIndex-1,SYMBOLDATA.sequenceLength,SYMBOLDATA.filename);
fprintf('Total penalty: %.3f bits (%.4f bits per symbol).\n',SYMBOLDATA.totalPenaltyInBits,SYMBOLDATA.totalPenaltyInBits/SYMBOLDATA.sequenceLength);
fprintf('Correctly predicted %.3f%% of symbols.\n',100*SYMBOLDATA.correctPredictions/SYMBOLDATA.sequenceLength);

pctileBoundaries = [0 1 5 10 20 30 40 50 60 70 80 90 100];

for pctileIndex = 1:length(pctileBoundaries)-1
    if pctileBoundaries(pctileIndex) == 0
        % Include probabilities that are exactly 0.
        a = sum((SYMBOLDATA.winnerProbabilities >= pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.winnerProbabilities <= pctileBoundaries(pctileIndex+1)/100));
        b = sum((SYMBOLDATA.loserProbabilities(:) >= pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.loserProbabilities(:) <= pctileBoundaries(pctileIndex+1)/100));
    else
        a = sum((SYMBOLDATA.winnerProbabilities > pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.winnerProbabilities <= pctileBoundaries(pctileIndex+1)/100));
        b = sum((SYMBOLDATA.loserProbabilities(:) > pctileBoundaries(pctileIndex)/100) & (SYMBOLDATA.loserProbabilities(:) <= pctileBoundaries(pctileIndex+1)/100));
    end
    if a+b > 0
        fprintf('%d probabilities forecasted between %d%% and %d%%; actual occurrence rate %.2f%%.\n',a+b,pctileBoundaries(pctileIndex),pctileBoundaries(pctileIndex+1),100*a/(a+b));
    else
        fprintf('0 probabilities forecasted between %d%% and %d%%.\n',pctileBoundaries(pctileIndex),pctileBoundaries(pctileIndex+1));
    end
end
fprintf('\n');