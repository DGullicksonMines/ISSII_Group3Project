% Dependencies:
%   Deep Learning Toolbox
%   Statistics and Machine Learning Toolbox
% Also consider using the Parallel Computing Toolbox if you've got a
% CUDA-compatible GPU for gpu-accelerated training

% Helpful source for understanding:
% https://www.mathworks.com/help/deeplearning/ug/multistep-neural-network-prediction.html

close all;

seq_to_load = "heart1";

% Load the sequence into a variable
l_seq = seq_to_load;
seq = load(sprintf('sequence_%s_train.mat', l_seq));
seq = seq.sequence;
seq_len = length(seq);

% Create input data
inputs = cell(1, seq_len);
for i = 1:seq_len
    inputs{i} = zeros(9, 1);
    inputs{i}(seq(i)) = 1;
%     inputs{i} = seq(i);
end
% Each slice, taken by inputs{n}, represents all 9 channels

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Nonlinear Autoregressive Network
%NOTE feedbackDelays controls how many past points are used for
%     prediction.
feedbackDelays = 1:2;
hiddenLayers = [9 50 100 50 9];
net = narnet(feedbackDelays, hiddenLayers, 'open', trainFcn);

% Set training parameters
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;

% Train the network
net.trainParam.showWindow = false;
net.trainparam.showCommandLine = true;
net = train(net, inputs, inputs, {}, {});

% Now actually run it
sequenceLength = initializeSymbolMachine( ...
    sprintf('sequence_%s_test.mat', l_seq) ...
);
test_data = zeros(1, sequenceLength);
predicted_data = zeros(sequenceLength, 2);

% Make initial prediction
[actual_sym, ~] = symbolMachine(ones(9,1) ./ 9);
last_symbol = zeros(9,1);
last_symbol(actual_sym) = 1;

test_data(1) = actual_sym;
predicted_data(1, 1) = 0;
predicted_data(1, 2) = 0;

input_state = {};
layer_state = {};

for i = 2:sequenceLength
    % Make prediction
    [predicted, input_state, layer_state] = ...
        net({last_symbol}, input_state, layer_state);
    predicted = predicted{1};

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
% fill([1:sequenceLength sequenceLength:-1:1], ...
%     [predicted_data(:, 1).' fliplr(predicted_data(:, 2).')], ...
%     [0.5 0.5 0.5], 'DisplayName', 'Range')
plot(predicted_data(:, 1), "--", 'DisplayName', 'First Prediction');
% plot(predicted_data(:, 2), "--", 'DisplayName', 'Second Prediction');
plot(test_data, 'DisplayName', 'Actual');
hold off;


