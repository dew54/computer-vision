close all force
clear all

TestDatasetPath  = fullfile('test');
imageSize = [224 224 3];
divideby=255;

imdsTest = imageDatastore(TestDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(imread(x))/divideby;

%imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);
augimdsTest = augmentedImageDatastore(imageSize,imdsTest, 'ColorPreprocessing', 'gray2rgb');

cifar2TrainDatasetPath = fullfile('train');
imds = imageDatastore(cifar2TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%imds.ReadFcn = @(x)imresize(imread(x),[64 64]);

imds.ReadFcn = @(x)double(imread(x))/divideby;


quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');



imageAugmenter = imageDataAugmenter( ...
    'RandYReflection', true);%'RandXReflection', true, ...
    %'RandRotation',[-20,20], ...
    %'RandXTranslation',[-3 3], ...
    %'RandYTranslation',[-3 3], ...
    
augimds = augmentedImageDatastore(imageSize,imdsTrain, 'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'gray2rgb' );
%augimds.MiniBatchSize = 1024


augimdsval = augmentedImageDatastore(imageSize,imdsValidation, 'ColorPreprocessing', 'gray2rgb');
%augimdsval.MiniBatchSize = 1024

classes=unique(imds.Labels)
%%
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsval, ... 
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf,...
    'Verbose',false, ...
    'MiniBatchSize',64, ...
    'L2Regularization',0.0001, ...
    'LearnRateSchedule','piecewise', 'LearnRateDropPeriod',2 , 'LearnRateDropFactor',0.7 , ...
    'Plots','training-progress');


%% Freeze weights
net = resnet18;
lgraph = layerGraph(net)

toFreeze = (1:65)
layersToFreeze = net.Layers(toFreeze)



for i = 1 : length(toFreeze)
    if(isprop(layersToFreeze(i), 'WeightLearnRateFactor'))
        layersToFreeze(i).WeightLearnRateFactor = 0;
        layersToFreeze(i).Name = 'freeze_' + string(i)
        lgraph = replaceLayer(lgraph,net.Layers(toFreeze(i)).Name,layersToFreeze(i));
    end
end
output = net.Layers(end)
output.Classes = classes
lgraph = replaceLayer(lgraph,output.Name,output);

newLearnableLayer = fullyConnectedLayer(15, ...
        'Name','new_fc', 'WeightsInitializer', 'he');
lgraph = replaceLayer(lgraph,net.Layers(69).Name,newLearnableLayer);


%% Estrazione delle funzioni di attivazione come nuovo dataset
net = resnet18
layer = 'res5a';
train_samples = activations(net,augimds,layer,'OutputAs','rows');
test_samples = activations(net,augimdsTest,layer,'OutputAs','rows');

train_samples_labels = imdsTrain.Labels;
test_samples_labels = imdsTest.Labels;
%% Normalizzazione tra -1 ed 1  per train e test set
range = max(train_samples(:)) - min(train_samples(:))   
m01 = (train_samples - min(train_samples))/range
train_samples = 2 * m01 -1

rangeTest = max(test_samples(:)) - min(test_samples(:))
t01 = (test_samples - min(test_samples))/range
test_samples = 2 * t01 -1
%%


classifier = fitcecoc(train_samples,train_samples_labels);


YPred = predict(classifier,test_samples);

accuracy = sum(YPred == test_samples_labels)/numel(test_samples_labels)

figure

plotconfusion(test_samples_labels,YPred)


%%
net = trainNetwork(augimds, lgraph, options)

%% Assessment

% apply the network to the test set
YPredicted = classify(net,augimdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure

plotconfusion(YTest,YPredicted)


