% Read the Excel file and store the data as vectors
data = readtable('Data.xlsx');

% Optimization Parameters
window_size = 48;  % Number of rows for an observation
prediction_horizon = 1;  % Prediction horizon
numHiddenUnits = 4096;  % Number of hidden neurons
numInitializations = 2;  % Number of initializations to try

% Extract necessary columns
TotalProduction_MW = max(data{:, 4}, 0);  
Thermal_MW = max(data{:, 5}, 0);
Hydro_MW = max(data{:, 7}, 0);
Solar_MW = max(data{:, 9}, 0);
Wind_MW = max(data{:, 10}, 0);
BioEnergy_MW = max(data{:, 11}, 0);
Import_MW = max(data{:, 13}, 0);

DateTime = datetime(data{:, 3}, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ssXXX', 'TimeZone', 'Europe/Paris');
Hours = hour(DateTime);

% Transform the hour (0 to 23) with an offset
Hours_offset = Hours + 1;  % Add offset to ensure positive values

Hours_sin = sin(2 * pi * Hours_offset / 24);   
Hours_cos = cos(2 * pi * Hours_offset / 24);

% Create a matrix containing all variables
input_matrix = [TotalProduction_MW, Thermal_MW, Hydro_MW, ...
                Solar_MW, Wind_MW, BioEnergy_MW, Import_MW, ...
                Hours_sin, Hours_cos];

%% Input-Output with Preallocation and windowed data

numRows = size(input_matrix, 1);
numObservations = numRows - window_size - prediction_horizon;

% Preallocate matrices X and Y
X = zeros(numObservations, window_size * size(input_matrix, 2));  
Y = zeros(numObservations, size(input_matrix, 2) - 2);  % Outputs except sine/cosine

% Populate X and Y matrices
for i = 1:numObservations
    X(i, :) = reshape(input_matrix(i:i+window_size-1, :)', 1, []);  
    Y(i, :) = input_matrix(i+window_size+prediction_horizon-1, 1:end-2);  
end

% Replace NaNs with 0
X(isnan(X)) = 0;
Y(isnan(Y)) = 0;

% Prepare training and testing data
trainSize = round(0.8 * size(X, 1));
XTrain = X(1:trainSize, :);
YTrain = Y(1:trainSize, :);
XTest = X(trainSize+1:end, :);
YTest = Y(trainSize+1:end, :);

% ELM model parameters
numOutputs = size(YTrain, 2);  

best_rmse = inf;  % Initialize best RMSE
best_inputWeights = [];  % Initialize best input weights
best_bias = [];  % Initialize best bias
best_outputWeights = [];  % Initialize best output weights

% Test multiple initializations
for init = 1:numInitializations
    disp(['Initialization #', num2str(init)]);
    
    % Initialize hidden layer weights and bias
    inputWeights = rand(numHiddenUnits, size(XTrain, 2));
    bias = rand(numHiddenUnits, 1);
    
    % Compute hidden layer output
    H = max(0, XTrain * inputWeights' + bias');  % ReLU activation function
    
    % Compute output weights using linear regression
    outputWeights = pinv(H) * YTrain;  % Corrected regression
    
    % Make predictions on test set
    H_test = max(0, XTest * inputWeights' + bias');  
    YPred = H_test * outputWeights;  % Unnormalized predictions
    
    % Ensure predicted values are non-negative
    YPred = max(YPred, 0); 

    % Calculate error metrics
    rmse_values = zeros(numOutputs, 1);
    for j = 1:numOutputs
        y_true = YTest(:, j);
        y_pred = YPred(:, j);
        
        rmse = sqrt(mean((y_pred - y_true).^2));  % RMSE
        rmse_values(j) = rmse;  % Store RMSE for each output
    end
    
    % Compute mean RMSE across all outputs
    mean_rmse = mean(rmse_values);
    
    % Keep the best initialization
    if mean_rmse < best_rmse
        best_rmse = mean_rmse;
        best_inputWeights = inputWeights;
        best_bias = bias;
        best_outputWeights = outputWeights;
    end
end

disp(['Best RMSE: ', num2str(best_rmse)]);

% Final predictions with best weights
H_final = max(0, XTest * best_inputWeights' + best_bias');  
YPred_final = H_final * best_outputWeights;  % Unnormalized predictions

% Ensure predicted values are non-negative
YPred_final = max(YPred_final, 0); 

% Final error metrics calculations
n = size(YTest, 1);  % Number of test samples
numOutputs = size(YTest, 2);  % Number of output variables
outputNames = {'Total_MW', 'Thermal_MW', 'Hydro_MW', ...
               'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW'};

nRMSE = zeros(numOutputs, 1);
nMAE = zeros(numOutputs, 1);
nMBE = zeros(numOutputs, 1);
R2 = zeros(numOutputs, 1);
gain_nRMSE = zeros(numOutputs, 1); % Preallocate gain_nRMSE for each output

% Persistence model for comparison
YPersistence = zeros(size(YTest));
for i = 1:size(YTest, 1)
    if i - prediction_horizon > 0
        YPersistence(i, :) = YTest(i - prediction_horizon, :);
    else
        YPersistence(i, :) = 0; % Replace NaNs with 0 to handle the initial condition
    end
end

% Calculate metrics for predictions and persistence
for j = 1:numOutputs
    y_true = YTest(:, j);
    y_pred = YPred_final(:, j);
    y_persist = YPersistence(:, j);
    
    % Calculate error metrics
    rmse = sqrt(mean((y_pred - y_true).^2));  % RMSE
    mae = mean(abs(y_pred - y_true));  % MAE
    mbe = mean(y_pred - y_true);  % MBE
    
    mean_y = mean(y_true);
    nRMSE(j) = rmse / mean_y;  % Normalized RMSE
    nMAE(j) = mae / mean_y;  % Normalized MAE
    nMBE(j) = mbe / mean_y;
    
    % Persistence error metrics
    rmse_persist = sqrt(mean((y_persist - y_true).^2));  % RMSE of persistence
    nRMSE_persist = rmse_persist / mean_y;  % Normalized RMSE of persistence
    
    % Gain of nRMSE for each output variable
    gain_nRMSE(j) = (nRMSE_persist - nRMSE(j)) / nRMSE_persist; % Store gain for each output
    
    % R² calculation
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    R2(j) = 1 - (ss_res / ss_tot);
    
    % Display metrics for each output variable
    disp(['--- Metrics for ', outputNames{j}, ' ---']);
    disp(['nRMSE: ', num2str(nRMSE(j)), ' | Gain: ', num2str(gain_nRMSE(j))]);
    disp(['nMAE: ', num2str(nMAE(j))]);
    disp(['nMBE: ', num2str(nMBE(j))]);
    disp(['R²: ', num2str(R2(j))]);
end

% Plot predictions for each output variable
figure; % Create a new figure for plotting
% Initialize an array to hold the axes handles for linking
ax = gobjects(numOutputs, 1); 

for j = 1:numOutputs
    ax(j) = subplot(numOutputs, 1, j); % Create a subplot for each output
    plot(YTest(:, j), 'b-', 'LineWidth', 1.5); hold on;  % Plot actual values in blue
    plot(YPred_final(:, j), 'r--', 'LineWidth', 1.5);  % Plot predicted values in red
    xlim([1, 100]);
        
    % Set title with various metrics including R²
    title({['nRMSE: ', num2str(nRMSE(j), '%.4f'), ', Gain: ', num2str(gain_nRMSE(j), '%.4f'), ...
            ', R²: ', num2str(R2(j), '%.4f'), ...
            ', nMAE: ', num2str(nMAE(j), '%.4f'), ', nMBE: ', num2str(nMBE(j), '%.4f')]}, ...
           'FontSize', 8, 'Interpreter', 'none', 'LineWidth', 1.5);
    
    xlabel('Time'); % Label for x-axis
    ylabel(outputNames{j}); % Label for y-axis
    legend('Actual Values', 'Predicted Values'); % Add legend
    grid on; % Enable grid
    
    % Link the x-axes of all subplots
    if j == 1
        linkaxes(ax(1:j), 'x'); % Link axes for the first subplot
    else
        linkaxes(ax(1:j), 'x'); % Link axes for all previous subplots
    end
end

% Adjust subplot spacing and add a main title
sgtitle(['Forecasts of Output Energy Variables in Corsica For horizon ', num2str(prediction_horizon),'h']); % Main title for the figure
