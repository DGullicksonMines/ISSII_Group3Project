%NOTE This predicts using info from the *test* sequences.
%NOTE It should not be used for results of any kind.

seq_to_load = "nonuniform";

% Load the sequence into a variable
test_seq = load(sprintf('sequence_%s_test.mat', seq_to_load));
test_seq = struct2table(test_seq);

sequence_test = test_seq{:, 'sequence'};

% Calculate the frequency of each value in sequence_test
value_counts = tabulate(sequence_test);

% ---== Test with symbol machine ==---
sequenceLength = initializeSymbolMachine( ...
    sprintf('sequence_%s_test.mat', seq_to_load) ...
);
for i = 1:sequenceLength
    [~, ~] = symbolMachine(value_counts(:, 3)/100);

    % Print out a status every so often
    if mod(i, 1000) == 0
        fprintf("%.4f%% complete.\n", 100*i/sequenceLength);
    end
end

reportSymbolMachine;