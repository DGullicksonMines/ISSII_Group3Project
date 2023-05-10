% Dependencies:
%   Deep Learning Toolbox
%   Statistics and Machine Learning Toolbox
% Also consider using the Parallel Computing Toolbox if you've got a
% CUDA-compatible GPU for gpu-accelerated training

% Helpful source for understanding:
% https://www.mathworks.com/help/deeplearning/ug/multistep-neural-network-prediction.html

close all;

seq_to_load = "heart2";
multichannel = false;
% Only for multichannel = false
stdev = 0.5;
% stdev = 'auto';

% Load the sequence into a variable
l_seq = seq_to_load;
seq = load(sprintf('sequence_%s_train.mat', l_seq));
seq = seq.sequence;
seq_len = length(seq);

% Create input data
inputs = cell(1, seq_len);
for i = 1:seq_len
    if multichannel
        inputs{i} = zeros(9, 1);
        inputs{i}(seq(i)) = 1;
    else
        inputs{i} = seq(i);
    end
end
% Each slice, taken by inputs{n}, represents all 9 channels

% Set standard deviation used
if ~multichannel && strcmp(stdev, 'auto')
    disp Auto
    stdev = std(cell2mat(inputs));
end

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
hiddenLayers = [10];
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
if multichannel
    last_symbol = zeros(9, 1);
    last_symbol(actual_sym) = 1;
else
    last_symbol = actual_sym;
end

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
    if ~multichannel
        predicted = normpdf(1:9, predicted, stdev);
    end

    % Make sure the prediciton is normalized, positive, and reasonable
    predicted = max(predicted, 0.0001);
    predicted = predicted/sum(predicted);

    [actual_sym, ~] = symbolMachine(predicted);
    if multichannel
        last_symbol = zeros(9, 1);
        last_symbol(actual_sym) = 1;
    else
        last_symbol = actual_sym;
    end

    test_data(i) = actual_sym;
    [~, sorted] = sort(predicted, "descend");
    predicted_data(i, 1) = sorted(1);
    predicted_data(i, 2) = sorted(2);

    % Print out a status every so often
    if mod(i, 1000) == 0
        fprintf("%.4f%% complete.\n", 100*i/sequenceLength);
    end
end

reportSymbolMachine;

% % Plot actual vs predicted
% figure;
% title("Actual vs Predicted");
% xlabel("Time");
% legend();
% hold on;
% % fill([1:sequenceLength sequenceLength:-1:1], ...
% %     [predicted_data(:, 1).' fliplr(predicted_data(:, 2).')], ...
% %     [0.5 0.5 0.5], 'DisplayName', 'Range')
% plot(predicted_data(:, 1), "--", 'DisplayName', 'First Prediction');
% % plot(predicted_data(:, 2), "--", 'DisplayName', 'Second Prediction');
% plot(test_data, 'DisplayName', 'Actual');
% hold off;


