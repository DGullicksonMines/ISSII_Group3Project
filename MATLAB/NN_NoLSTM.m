% Dependencies:
%   Deep Learning Toolbox
%   Statistics and Machine Learning Toolbox
% Also consider using the Parallel Computing Toolbox if you've got a
% CUDA-compatible GPU for gpu-accelerated training

close all;

seq_to_load = "Hawaiian";

% Train the network only once unless something has changed
% to force a retrain, delete lstm_net from your workspace
if ~exist("lstm_net", "var") || ~exist("l_seq", "var") || seq_to_load ~= l_seq

    % Load the sequence into a variable
    l_seq = seq_to_load;
    seq = load(sprintf('sequence_%s_train.mat', l_seq));
    seq = seq.sequence;

    % Split this sequence into individual training sequences
    train_seq_len = 40;
    train_seqs = floorDiv(length(seq) - 1, train_seq_len);

    inputs = cell(train_seqs, 1);
    responses = cell(train_seqs, 1);

    % For each training sequence and position, set a channel to 1
    seq_index = 1;
    for i = 1:train_seqs
        inputs{i} = zeros(9, train_seq_len);
        responses{i} = zeros(9, train_seq_len);

        for j = 1:train_seq_len
            inputs{i}(seq(seq_index), j) = 1;
            responses{i}(seq(seq_index + 1), j) = 1;
            seq_index = seq_index + 1;
        end
    end
    % Each slice, taken by arr(n, :, :), represents all 9 channels

    % Create and run network 
    layers = [
        sequenceInputLayer(9)
        fullyConnectedLayer(1000)
        fullyConnectedLayer(500)
        fullyConnectedLayer(100)
        fullyConnectedLayer(5)
        fullyConnectedLayer(100)
        fullyConnectedLayer(9)
        regressionLayer
    ];
    
    options = trainingOptions("adam", ...
        MaxEpochs = 250, ...
        Shuffle = "every-epoch", ...
        Plots = "training-progress", ...
        Verbose = 0 ...
    );
    
    lstm_net = trainNetwork(inputs, responses, layers, options);
    disp("Trained new network");
end

% Now actually run it
sequenceLength = initializeSymbolMachine( ...
    sprintf('sequence_%s_test.mat', l_seq) ...
);
test_data = zeros(sequenceLength, 1);
predicted_data = zeros(sequenceLength, 2);

lstm_net = resetState(lstm_net);
last_symbol = ones(9,1) ./ 9;

for i = 1:sequenceLength
    [lstm_net, predicted] = predictAndUpdateState(lstm_net, last_symbol);

    % Make sure the prediciton is normalized, positive, and reasonable
    predicted = max(predicted, 0.0001);
    predicted = predicted/sum(predicted);

    [actual_sym, ~] = symbolMachine(predicted);
    last_symbol = zeros(9,1);
    last_symbol(actual_sym) = 1;

    test_data(i) = actual_sym;
    [~, sorted] = sort(predicted, "descend");
    predicted_data(i, 1) = sorted(1);
    predicted_data(i, 2) = sorted(2);

    % Print out a heartbeat every x steps
    if mod(i, 100) == 0
        disp("Completed " + i + " iterations");
    end
end

reportSymbolMachine;

% Plot actual vs predicted
figure;
title("Actual vs Predicted");
xlabel("Time");
legend();
hold on;
plt1 = plot(test_data, 'DisplayName', 'Actual');
plt2 = plot(predicted_data(:, 1), "--", 'DisplayName', 'First Prediction');
plt3 = plot(predicted_data(:, 2), "--", 'DisplayName', 'Second Prediction');
hold off;