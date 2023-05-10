seq_to_load = "uniform";

% Load the sequence into a variable
test_seq = load(sprintf('sequence_%s_test.mat', seq_to_load));
test_seq = struct2table(test_seq);

sequence_test = test_seq{:, 'sequence'};

% Calculate the frequency of each value in sequence_test
value_counts = tabulate(sequence_test);

% Calculate the entropy of each symbol
probs = value_counts(:, 3)/100;
probs = probs(probs ~= 0);
entropy = -sum(log2(probs).*probs);

% Find the most common value
most_common_value = value_counts(value_counts(:, 2) == max(value_counts(:, 2)), 1);

% Calculate the accuracy of the mcv model
mcv_accuracy = sum(most_common_value == sequence_test) / numel(sequence_test);

% Display stats
fprintf('Baseline bits per symbol: %.4f\n', entropy);
fprintf('Baseline Accuracy: %.2f%%\n', mcv_accuracy * 100);