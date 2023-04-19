% Copyright (C) 2023 Dawson J. Gullickson All rights reserved.

% Used for reference:
% https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html

% Requires the following Add-ons:
% - Deep Learning Toolbox
% - Statistics and Machine Learning Toolbox

close all;

sequence_to_load = "heart2";

% Only train if lstm_net doesn't yet exist
% OR if the sequence was changed
if ~exist("lstm_net", "var") ...
|| ~exist("loaded_seq", "var") ...
|| sequence_to_load ~= loaded_seq
    % Load sequence
    loaded_seq = sequence_to_load;
    sequenceLength = initializeSymbolMachine( ...
        sprintf('sequence_%s_train.mat', loaded_seq) ...
    );
    
    % Get a sequence of symbols
    probs = ones(9)/9;
    data = zeros(sequenceLength);
    for i = 1:sequenceLength
        [symbol, ~] = symbolMachine(probs);
        data(i) = symbol;
    end
    
    % Create vector of inputs and responses
    num_sequences = 20;
    seq_len = floor((size(data)-1)/num_sequences);
    inputs = cell(num_sequences, 1);
    responses = cell(num_sequences, 1);
    for seq = 1:num_sequences
        fro = seq_len*(seq-1)+1;
        til = seq_len*seq;
        inputs{seq} = data(fro:til);
        responses{seq} = data(fro+1:til+1);
    end
    
    % Create and run network
    layers = [
        sequenceInputLayer(1)
        lstmLayer(100)
        fullyConnectedLayer(1)
        regressionLayer
    ];
    
    % Settings for training the network
    % NOTE MaxEpochs controls the number of "rounds" for training
    options = trainingOptions( ...
        "adam", ...
        MaxEpochs=500, ...
        Shuffle="every-epoch", ...
        Plots="training-progress", ...
        Verbose=0 ...
    );
    
    % Train the network
    lstm_net = trainNetwork(inputs, responses, layers, options);
end

% Load testing data
sequenceLength = initializeSymbolMachine( ...
    sprintf('sequence_%s_test.mat', loaded_seq) ...
);

% Test network
test_data = zeros(sequenceLength);
predicted_data = zeros(sequenceLength);

lstm_net = resetState(lstm_net);
last_symbol = 0;
% count = 0;
for i = 1:sequenceLength
    [lstm_net, predicted] = predictAndUpdateState(lstm_net, last_symbol);
%     probs = normpdf(1:9, predicted, 1);
%     probs = probs/sum(probs);
    closest = max(1, min(9, round(predicted)));
%     probs = zeros(9);
%     probs(closest) = 1;
    probs = normpdf(1:9, closest, 0.5);
    probs = probs/sum(probs);

    [last_symbol, ~] = symbolMachine(probs);

    test_data(i) = last_symbol;
    predicted_data(i) = closest;
%     if last_symbol == closest
%         count = count + 1;
%     end
end
reportSymbolMachine;

% fprintf("%d\n", 100*count/sequenceLength);

% Plot actual vs predicted
figure;
hold on;
plot(1:sequenceLength, test_data);
plot(1:sequenceLength, predicted_data, "--");
hold off;
title("Actual vs Predicted");
xlabel("Time");
legend(["Actual", "Predicted"]);