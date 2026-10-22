%% 📌 Charger les Données et les Transformer en Séries Temporelles
data = readtable('Data.xlsx');
tic

% Paramètres
window_size = 48;  % Nombre de pas de temps pour une observation
prediction_horizon = 5;  % Horizon de prédiction
numHiddenUnits = 500;  % Même nombre de neurones cachés que ELM

% Extraire les variables énergétiques
TotalProduction_MW = max(data{:, 4}, 0);
Thermal_MW = max(data{:, 5}, 0);
Hydro_MW = max(data{:, 7}, 0);
Solar_MW = max(data{:, 9}, 0);
Wind_MW = max(data{:, 10}, 0);
BioEnergy_MW = max(data{:, 11}, 0);
Import_MW = max(data{:, 13}, 0);

% Conversion des dates
DateTime = datetime(data{:, 3}, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ssXXX', 'TimeZone', 'Europe/Paris');
Hours = hour(DateTime);

% Transformation sinusoïdale des heures
Hours_sin = sin(2 * pi * (Hours + 1) / 24);
Hours_cos = cos(2 * pi * (Hours + 1) / 24);

% Matrice d'entrée
input_matrix = [TotalProduction_MW, Thermal_MW, Hydro_MW, ...
                Solar_MW, Wind_MW, BioEnergy_MW, Import_MW, ...
                Hours_sin, Hours_cos];

% Dimensions
[numRows, numFeatures] = size(input_matrix);
numObservations = numRows - window_size - prediction_horizon;

%% Préparation des données pour LSTM (Format 3D : [features, timeSteps, samples])
X = zeros(numFeatures, window_size, numObservations);
Y = zeros(numObservations, numFeatures - 2);  % On exclut les sinusoïdes des sorties

for i = 1:numObservations
    X(:, :, i) = input_matrix(i:i+window_size-1, :)';  
    Y(i, :) = input_matrix(i+window_size+prediction_horizon-1, 1:end-2);
end

% Séparation en ensemble d'entraînement et test
trainSize = round(0.8 * numObservations);
XTrain = X(:, :, 1:trainSize);
YTrain = Y(1:trainSize, :);
XTest = X(:, :, trainSize+1:end);
YTest = Y(trainSize+1:end, :);

XTrain = squeeze(num2cell(XTrain, [1 2]));  % Convertir en cellule [batchSize × 1]
XTest = squeeze(num2cell(XTest, [1 2]));    % Même correction pour le test

disp("Taille finale de XTrain avant trainNetwork : " + mat2str(size(XTrain)));  
disp("Taille finale de premier élément de XTrain : " + mat2str(size(XTrain{1})));  

assert(iscell(XTrain), "ERREUR : XTrain doit être une cellule !");
assert(size(XTrain{1},1) == 9, "ERREUR : Chaque séquence de XTrain doit avoir 9 features !");
assert(size(XTrain{1},2) == 48, "ERREUR : Chaque séquence de XTrain doit avoir 48 time steps !");

% Conversion de YTrain et YTest en matrice `double`
YTrain = double(YTrain);
YTest = double(YTest);


%% Définition du Réseau LSTM
layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')  % Entrée avec 9 variables
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm') % 4096 neurones LSTM
    fullyConnectedLayer(size(YTrain, 2), 'Name', 'fc') % 7 sorties
    regressionLayer('Name', 'output')]; % Régression

% Options d'entraînement
options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots', 'training-progress');

%% Entraînement du Modèle
net = trainNetwork(XTrain, YTrain, layers, options);

%% Prédictions sur les Données de Test
YPred = predict(net, XTest);
YPred = max(YPred, 0); % Assurer que les prédictions sont positives

%% Calcul des Métriques d'Erreur
n = size(YTest, 1);
numOutputs = size(YTest, 2);
outputNames = {'Total_MW', 'Thermal_MW', 'Hydro_MW', ...
               'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW'};

nRMSE = zeros(numOutputs, 1);
nMAE = zeros(numOutputs, 1);
nMBE = zeros(numOutputs, 1);
R2 = zeros(numOutputs, 1);
gain_nRMSE = zeros(numOutputs, 1);

% Modèle de persistance pour comparaison
YPersistence = zeros(size(YTest));
for i = 1:size(YTest, 1)
    if i - prediction_horizon > 0
        YPersistence(i, :) = YTest(i - prediction_horizon, :);
    else
        YPersistence(i, :) = 0; 
    end
end

% Calcul des métriques
for j = 1:numOutputs
    y_true = YTest(:, j);
    y_pred = YPred(:, j);
    y_persist = YPersistence(:, j);
    
    % Erreurs
    rmse = sqrt(mean((y_pred - y_true).^2));
    mae = mean(abs(y_pred - y_true));
    mbe = mean(y_pred - y_true);
    
    mean_y = mean(y_true);
    nRMSE(j) = rmse / mean_y;
    nMAE(j) = mae / mean_y;
    nMBE(j) = mbe / mean_y;
    
    % Comparaison avec modèle de persistance
    rmse_persist = sqrt(mean((y_persist - y_true).^2));
    nRMSE_persist = rmse_persist / mean_y;
    gain_nRMSE(j) = (nRMSE_persist - nRMSE(j)) / nRMSE_persist;
    
    % Coefficient de détermination R²
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    R2(j) = 1 - (ss_res / ss_tot);
    
    % Affichage
    disp(['--- ', outputNames{j}, ' ---']);
    disp(['nRMSE: ', num2str(nRMSE(j)), ' | Gain: ', num2str(gain_nRMSE(j))]);
    disp(['nMAE: ', num2str(nMAE(j))]);
    disp(['nMBE: ', num2str(nMBE(j))]);
    disp(['R²: ', num2str(R2(j))]);
end

%% Visualisation des Prédictions
figure;
for j = 1:numOutputs
    subplot(numOutputs, 1, j);
    plot(YTest(:, j), 'b-', 'LineWidth', 1.5); hold on;
    plot(YPred(:, j), 'r--', 'LineWidth', 1.5);
    title(outputNames{j});
    xlabel('Temps');
    ylabel(outputNames{j});
    legend('Réel', 'Prédit');
    grid on;
end
toc
sgtitle(['Prédictions LSTM pour Horizon ', num2str(prediction_horizon),'h']);
