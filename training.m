% Chargement du dataset
Dataset = imageDatastore("My-Dataset", 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Séparation des données en ensembles d'entraînement et de validation²
[Training_Dataset, Validation_Dataset] = splitEachLabel(Dataset, 7.0);

% Chargement du réseau pré-entraîné GoogleNet
net = googlenet;

% Analyse du réseau
analyzeNetwork(net);

% Extraction des informations sur le réseau
Input_Layer_Size = net.Layers(1).InputSize;
Layer_Graph = layerGraph(net);
Feature_Learner = net.Layers(142);
Output_Classifier = net.Layers(144);
Number_of_Classes = numel(categories(Training_Dataset.Labels));

% Création d'un nouveau "Feature Learner"
New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'Facial Feature Learner', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

% Création d'une nouvelle couche de classification
New_Classifier_Layer = classificationLayer("Name", 'Face Classifier');

% Remplacement des couches dans le graphe
Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);
Layer_Graph = replaceLayer(Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);

% Analyse du nouveau graphe du réseau
analyzeNetwork(Layer_Graph);

% Définition des paramètres d'augmentation d'image
Pixel_Range = [-30 30];
Scale_Range = [0.9 1.1];
Image_Augmenter = imageDataAugmenter("RandXReflection", true, ...
    'RandXTranslation', Pixel_Range, ...
    'RandYTranslation', Pixel_Range, ...
    'RandXScale', Scale_Range, ...
    'RandYScale', Scale_Range);

% Création des datastore augmentés pour l'entraînement et la validation
Augmented_Training_Image = augmentedImageDatastore(Input_Layer_Size(1:2), Training_Dataset, ...
    'DataAugmentation', Image_Augmenter);
Augmented_Validation_Image = augmentedImageDatastore(Input_Layer_Size(1:2), Validation_Dataset);

% Définition des options de formation
Size_of_Minibatch = 5;
Validation_Frequency = 1;
Training_Options = trainingOptions('sgdm', ...
    'MiniBatchSize', Size_of_Minibatch, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Augmented_Validation_Image, ...
    'ValidationFrequency', Validation_Frequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Entraînement du réseau
net = trainNetwork(Augmented_Training_Image, Layer_Graph, Training_Options);

% Prédictions sur l'ensemble de validation augmenté
[YPred, scores] = classify(net, Augmented_Validation_Image);

% Identifier les personnes non reconnues
threshold = 0.8;
unrecognized_indices = find(max(scores, [], 2) < threshold);

if ~isempty(unrecognized_indices)
    fprintf('Les personnes suivantes n\''ont pas t reconnues :\n');
    for idx = 1:length(unrecognized_indices)
        fprintf('Personne %d\n', unrecognized_indices(idx));
    end
else
    fprintf('Toutes les personnes ont été reconnues avec un score de confiance supérieur ou égal à %.2f.\n', threshold);
end
