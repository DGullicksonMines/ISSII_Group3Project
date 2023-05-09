% Dependencies:
%   Deep Learning Toolbox
%   Statistics and Machine Learning Toolbox
% Also consider using the Parallel Computing Toolbox if you've got a
% CUDA-compatible GPU for gpu-accelerated training

%TODO Could add more predictors, such as 2 steps back.

close all;

seq_to_load = "heart1";
num_taps = 2;
time_predictor = false;

% ---== Load Data ==---

% Load the sequence into a variable
train_seq = load(sprintf('sequence_%s_train.mat', seq_to_load));
train_seq = struct2table(train_seq);

% Load the sequence into a variable
test_seq = load(sprintf('sequence_%s_test.mat', seq_to_load));
test_seq = struct2table(test_seq);

% ---== Construct tables ==---

% Get the sequence data
sequence_data = train_seq{:, 'sequence'};
sequence_test_data = test_seq{:, 'sequence'};

% Add the time data
train_time = 1:length(sequence_data);
train_seq.time = train_time.';
test_time = 1:length(sequence_test_data);
test_seq.time = test_time.';

predictorNames = cell(1, num_taps);
for i=0:num_taps
    delay = num_taps-i;

    % Shift the sequence data by one position
    %TODO instead of adding nan, maybe use end-1?
    shifted_sequence_data = [sequence_data(1+i:end); zeros(i, 1) + nan];
    shifted_test_sequence_data = [sequence_test_data(1+i:end); zeros(i, 1) + nan];

    % Add the shifted sequence data as a new column in wind_train_data
    seq_name = sprintf('delayed_%i', delay);
    train_seq.(seq_name) = shifted_sequence_data;
    test_seq.(seq_name) = shifted_test_sequence_data;

    if delay ~= 0
        predictorNames{i+1} = seq_name;
    end
end

% % Create one-hot shifted sequence
% categorized_data = 1:9 == shifted_sequence_data;
% train_seq.categorized_sequence = categorized_data;
% categorized_test_data = 1:9 == shifted_test_sequence_data;
% train_seq.categorized_sequence = categorized_test_data;

% ---== Create a sequence per number ==---
train_seqs = cell(9, 1);
for i = 1:9
    train_seqs{i} = varfun(@(x) double(x == i), train_seq);
    train_seqs{i}.Properties.VariableNames = train_seq.Properties.VariableNames;
end

% ---== Train Model ==---
if time_predictor
    predictorNames{end+1} = 'time';
end
responseName = 'delayed_0';

models = cell(9, 1);
for i = 1:9
    [model, ~] = trainRegressionModel(train_seqs{i}, predictorNames, responseName);
    models{i} = model;
end

% % ---== Test Model ==---
% yfit = model.predictFcn(test_seq);
% 
% % ---== Analyze Accuracy ==---
% % Code first rounds the predicted values in yfit1 to the nearest integer using the round function. It then gets the sequence data from wind_test_data and 
% % compares the rounded predicted values to the actual test data. The accuracy is calculated by counting the number of times that the predicted value 
% % matches the actual value, and dividing by the total number of elements in the sequence
% 
% % Round the predicted values to the nearest integer
% yfit_rounded = round(yfit);
% 
% % Get the sequence data from wind_test_data
% sequence_test = test_seq{:, 'sequence'};
% sequence_test_matched = sequence_test(num_taps+1:end);
% 
% % Compare the predicted values to the actual test data
% accuracy = sum(yfit_rounded(1:end-num_taps) == sequence_test_matched) / numel(sequence_test_matched);
% 
% % Display the accuracy as a percentage
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);
% 
% Compare the accuracy of your model against a baseline model that always predicts the most common 
% value in the test set, you can calculate the frequency of each value in sequence_test using the tabulate function

% Calculate the frequency of each value in sequence_test
% For each unique value in sequence_test, the tabulate function shows the number of instances and percentage of that 
% value in sequence_test
value_counts = tabulate(sequence_test);

% Find the most common value
most_common_value = value_counts(value_counts(:, 2) == max(value_counts(:, 2)), 1);

% Calculate the accuracy of the baseline model
baseline_accuracy = sum(most_common_value == sequence_test) / numel(sequence_test);

% Display the baseline accuracy as a percentage
fprintf('Baseline accuracy: %.2f%%\n', baseline_accuracy * 100);

% % ---== Results ==---
% % Create a new table with the rounded predicted values and sequence_test
% results_table = table(yfit, yfit_rounded, sequence_test);
% 
% % Rename the variable names in the table
% results_table.Properties.VariableNames = {'Yfit Sequence', 'Predicted_Sequence', 'Actual_Sequence'};

% ---== Test with symbol machine ==---
sequenceLength = initializeSymbolMachine( ...
    sprintf('sequence_%s_test.mat', seq_to_load) ...
);

running_seq = {};
running_seq.time = (1:sequenceLength).';
for delay = 0:num_taps
    seq_name = sprintf('delayed_%i', delay);
    running_seq.(seq_name) = zeros(sequenceLength, 1) + nan;
end
running_seq = struct2table(running_seq);
predicted_seq = zeros(sequenceLength, 1);
for i = 1:sequenceLength
    % Make predictions
    predicted = zeros(9, 1);
    for j = 1:9
        categorized_seq = varfun(@(x) double(x == j), running_seq);
        categorized_seq.Properties.VariableNames = running_seq.Properties.VariableNames;
        prediction = models{j}.predictFcn(categorized_seq);
        predicted(j) = prediction(max(i-num_taps, 1));
    end

    % Make sure the prediciton is normalized, positive, and reasonable
    predicted = max(predicted, 0.0001);
    predicted = predicted/sum(predicted);
    [~, argmax] = max(predicted);
    predicted_seq(i) = argmax;

    [actual_sym, ~] = symbolMachine(predicted);
    for j = 0:num_taps
        if i > j
            delay = num_taps - j;
            seq_name = sprintf('delayed_%i', delay);
            running_seq.(seq_name)(i-j) = actual_sym;
        end
    end
end

reportSymbolMachine;

% ---== Display graph ==---
figure;
title("Actual vs Predicted");
xlabel("Time");
hold on;
plot(running_seq.(sprintf('delayed_%i', num_taps)), 'DisplayName', 'Actual');
plot(predicted_seq, "--", 'DisplayName', 'Predicted');
hold off;
legend;

% ---== Caluclate bit penalty/% correct ==---
global SYMBOLDATA;
BPpPC = SYMBOLDATA.totalPenaltyInBits/SYMBOLDATA.correctPredictions;
fprintf('---\nBit penalty per %% correct: %.2f\n', BPpPC);

% ---== Regression Learner App PARTIALLY-Generated Training Function ==---
function [trainedModel, validationRMSE] = trainRegressionModel(trainingData, predictorNames, responseName)
    % [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
    % Returns a trained regression model and its RMSE. This code recreates the
    % model trained in Regression Learner app. Use the generated code to
    % automate training the same model with new data, or to learn how to
    % programmatically train models.
    %
    %  Input:
    %      trainingData: A table containing the same predictor and response
    %       columns as those imported into the app.
    %
    %  Output:
    %      trainedModel: A struct containing the trained regression model. The
    %       struct contains various fields with information about the trained
    %       model.
    %
    %      trainedModel.predictFcn: A function to make predictions on new data.
    %
    %      validationRMSE: A double containing the RMSE. In the app, the Models
    %       pane displays the RMSE for each model.
    %
    % Use the code to train the model with new data. To retrain your model,
    % call the function from the command line with your original data or new
    % data as the input argument trainingData.
    %
    % For example, to retrain a regression model trained with the original data
    % set T, enter:
    %   [trainedModel, validationRMSE] = trainRegressionModel(T)
    %
    % To make predictions with the returned 'trainedModel' on new data T2, use
    %   yfit = trainedModel.predictFcn(T2)
    %
    % T2 must be a table containing at least the same predictor columns as used
    % during training. For details, enter:
    %   trainedModel.HowToPredict
    
    % Auto-generated by MATLAB on 08-May-2023 11:51:34
    
    
    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    inputTable = trainingData;
    predictors = inputTable(:, predictorNames);
    response = inputTable.(responseName);
%     isCategoricalPredictor = [false];
    
    % Train a regression model
    % This code specifies all the model options and trains the model.
    regressionTree = fitrtree(...
        predictors, ...
        response, ...
        'MinLeafSize', 4, ...
        'Surrogate', 'off');
    
    % Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    treePredictFcn = @(x) predict(regressionTree, x);
    trainedModel.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));
    
    % Add additional fields to the result struct
    trainedModel.RequiredVariables = {'sequence'};
    trainedModel.RegressionTree = regressionTree;
    trainedModel.About = 'This struct is a trained model exported from Regression Learner R2022b.';
    trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');
    
    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
%     inputTable = trainingData;
%     predictorNames = {'sequence'};
%     predictors = inputTable(:, predictorNames);
%     response = inputTable.shifted_sequence;
%     isCategoricalPredictor = [false];
    
    % Perform cross-validation
    partitionedModel = crossval(trainedModel.RegressionTree, 'KFold', 5);
    
    % Compute validation predictions
    % validationPredictions = kfoldPredict(partitionedModel);
    
    % Compute validation RMSE
    validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
end