function [imds,layers,options] = experiment(params)



cifar2TrainDatasetPath = fullfile('C:\Users\dew54\OneDrive\Documenti\UniTs\ComputerVision\Project\trainingProject\train');
imds = imageDatastore(cifar2TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%imds.ReadFcn = @(x)imresize(imread(x),[64 64]);

divideby=255;
imds.ReadFcn = @(x)double(imread(x))/divideby;


quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

imageSize = [227 227 3];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true);
    %'RandXReflection', true);
    %'RandRotation',[-20,20], ...
    %'RandXTranslation',[-3 3], ...
    %'RandYTranslation',[-3 3], ...

classes=unique(imds.Labels)

imds = augmentedImageDatastore(imageSize,imdsTrain, 'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'gray2rgb' );
%augimds.MiniBatchSize = 1024


augimdsval = augmentedImageDatastore(imageSize,imdsValidation, 'ColorPreprocessing', 'gray2rgb');
%augimdsval.MiniBatchSize = 1024

options = trainingOptions(params.opt, ...
    'InitialLearnRate',params.learnRate, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsval, ... 
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf,...
    'Verbose',false, ...
    'MiniBatchSize',64, ...
    'L2Regularization',params.L2, ...
    'LearnRateSchedule','piecewise', 'LearnRateDropPeriod',3 , 'LearnRateDropFactor',0.5 , ...
    'Plots','training-progress');

alex = alexnet;
lgraph = layerGraph(alex.Layers)

toFreeze = [2, 6, 10, 12, 14, 17, 20]
layersToFreeze = alex.Layers(toFreeze)

for i = 1 : length(toFreeze)
    layersToFreeze(i).WeightLearnRateFactor = 0;
    layersToFreeze(i).Name = 'freeze_' + string(i)
    lgraph = replaceLayer(lgraph,alex.Layers(toFreeze(i)).Name,layersToFreeze(i));
end
output = alex.Layers(end)
output.Classes = classes
lgraph = replaceLayer(lgraph,output.Name,output);

newLearnableLayer = fullyConnectedLayer(15, ...
        'Name','new_fc');
layers = replaceLayer(lgraph,alex.Layers(23).Name,newLearnableLayer);



end