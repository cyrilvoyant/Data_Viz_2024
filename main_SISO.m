%% 📌 Charger les Données
data = readtable('Data.xlsx');
tic

% Paramètres
window_size = 48; %48 
prediction_horizon = 10;  
numHiddenUnits = 4096;  
numInitializations = 1;  

% Extraire les variables énergétiques
TotalProduction_MW = max(data{:, 4}, 0);
Thermal_MW = max(data{:, 5}, 0);
Hydro_MW = max(data{:, 7}, 0);
Solar_MW = max(data{:, 9}, 0);
Wind_MW = max(data{:, 10}, 0);
BioEnergy_MW = max(data{:, 11}, 0);
Import_MW = max(data{:, 13}, 0);

% Noms des variables énergétiques
outputNames = {'Total_MW', 'Thermal_MW', 'Hydro_MW', ...
               'Solar_MW', 'Wind_MW', 'BioEner_MW', 'Import_MW'};

% Stocker les variables sous forme de matrice
energy_data = [TotalProduction_MW, Thermal_MW, Hydro_MW, ...
               Solar_MW, Wind_MW, BioEnergy_MW, Import_MW];

% Nombre total d'observations et de sorties
numOutputs = size(energy_data, 2);
numRows = size(energy_data, 1);
numObservations = numRows - window_size - prediction_horizon;

%% Initialisation des Résultats SISO
YPred_SISO = zeros(numObservations - round(0.8 * numObservations), numOutputs);
nRMSE_SISO = zeros(numOutputs, 1);
nMAE_SISO = zeros(numOutputs, 1);

%% Boucle sur chaque variable pour créer un modèle ELM SISO
for j = 1:numOutputs
    disp(['⚡ Entraînement du modèle SISO pour : ', outputNames{j}]);

    % Préparer les entrées et sorties pour ce modèle SISO
    X_siso = zeros(numObservations, window_size);  
    Y_siso = zeros(numObservations, 1);  

    for i = 1:numObservations
        X_siso(i, :) = energy_data(i:i+window_size-1, j)';  
        Y_siso(i) = energy_data(i+window_size+prediction_horizon-1, j);
    end

    % Séparer en train/test
    trainSize = round(0.8 * numObservations);
    XTrain_siso = X_siso(1:trainSize, :);
    YTrain_siso = Y_siso(1:trainSize, :);
    XTest_siso = X_siso(trainSize+1:end, :);
    YTest_siso = Y_siso(trainSize+1:end, :);

    % Entraînement du modèle ELM SISO
    best_rmse_siso = inf;
    for init = 1:numInitializations
        % Initialisation des poids et biais
        inputWeights_siso = rand(numHiddenUnits, size(XTrain_siso, 2));
        bias_siso = rand(numHiddenUnits, 1);
        
        % Calcul de la couche cachée
        H_siso = max(0, XTrain_siso * inputWeights_siso' + bias_siso');  

        % Résolution pseudo-inverse
        lambda = 1e-6;
        outputWeights_siso = (H_siso' * H_siso + lambda * eye(size(H_siso,2))) \ (H_siso' * YTrain_siso);
        
        % Prédiction sur le test set
        H_test_siso = max(0, XTest_siso * inputWeights_siso' + bias_siso');
        YPred_siso = max(H_test_siso * outputWeights_siso, 0);
        
        % Calcul de l'erreur RMSE
        rmse_siso = sqrt(mean((YPred_siso - YTest_siso).^2));
        if rmse_siso < best_rmse_siso
            best_rmse_siso = rmse_siso;
            best_inputWeights_siso = inputWeights_siso;
            best_bias_siso = bias_siso;
            best_outputWeights_siso = outputWeights_siso;
        end
    end

    % Prédiction finale du modèle SISO
    H_final_siso = max(0, XTest_siso * best_inputWeights_siso' + best_bias_siso');
    YPred_SISO(:, j) = max(H_final_siso * best_outputWeights_siso(:), 0);

    % Calcul des métriques d'erreur
    nRMSE_SISO(j) = best_rmse_siso / mean(YTest_siso);
    nMAE_SISO(j) = mean(abs(YPred_SISO(:, j) - YTest_siso)) / mean(YTest_siso);
end

%% Affichage des résultats sous forme de tableau
disp("📊 Comparaison des performances SISO :");
result_table = table(outputNames', nRMSE_SISO, nMAE_SISO, ...
                     'VariableNames', {'Variable', 'nRMSE_SISO', 'nMAE_SISO'});
disp(result_table);

%% 📌 Visualisation des Prédictions SISO
% figure;
% for j = 1:numOutputs
%     subplot(numOutputs, 1, j);
%     plot(YTest_siso, 'b-', 'LineWidth', 1.5); hold on;
%     plot(YPred_SISO(:, j), 'r--', 'LineWidth', 1.5);
%     title(['Prédiction SISO - ', outputNames{j}]);
%     xlabel('Temps');
%     ylabel(outputNames{j});
%     legend('Réel', 'ELM SISO');
%     grid on;
% end

toc
