
close all force
clear all
%%

cifar2TrainDatasetPath = fullfile('train');
imds = imageDatastore(cifar2TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%imds.ReadFcn = @(x)imresize(imread(x),[64 64]);

divideby=255;
imds.ReadFcn = @(x)double(imread(x))/divideby;


quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');


imageSize = [64 64 1];
%%
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true);
    %'RandXReflection', true);
    %'RandRotation',[-20,20], ...
    %'RandXTranslation',[-3 3], ...
    %'RandYTranslation',[-3 3], ...
    
augimds = augmentedImageDatastore(imageSize,imdsTrain, 'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'rgb2gray');

%%
augimds = augmentedImageDatastore(imageSize,imdsTrain, 'ColorPreprocessing', 'rgb2gray');



%%
augimdsval = augmentedImageDatastore(imageSize,imdsValidation);

classes=unique(imds.Labels)


layers = [
    imageInputLayer([64 64 1],'Name','input') % 
    %dropoutLayer(.25,'Name','dropout_0')

    
    convolution2dLayer(3,8,'Padding','same', 'Stride',1, 'Name','conv_1','WeightsInitializer', @(sz) randn(sz) * 0.1, 'BiasInitializer', @(sz) rand(sz) * 0)
    %convolution2dLayer(3,8,'Padding','same', 'Stride',1, 'Name','conv_1','WeightsInitializer', 'he', 'BiasInitializer', @(sz) rand(sz) * 0)
    %convolution2dLayer(3,8,'Padding','same', 'Stride',1, 'Name','conv_1', 'BiasInitializer', @(sz) rand(sz) * 0)
    
    %batchNormalizationLayer('Name', 'norm1')
    
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    %dropoutLayer(.25,'Name','dropout_1')

    %convolution2dLayer(5,16,'Padding','same', 'Stride',1, 'Name','conv_2','WeightsInitializer', @(sz) randn(sz) * 0.1, 'BiasInitializer', @(sz) rand(sz) * 0)
    %convolution2dLayer(3,16,'Padding','same', 'Stride',1, 'Name','conv_2', 'BiasInitializer', @(sz) rand(sz) * 0)
    
    batchNormalizationLayer('Name', 'norm2')
    
    reluLayer('Name','relu_2')    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2') 

    %dropoutLayer(.25,'Name','dropout_1')

    convolution2dLayer(7,32,'Padding','same','Stride',1, 'Name','conv_3','WeightsInitializer',@(sz) randn(sz) * 0.1, 'BiasInitializer', @(sz) rand(sz) * 0)
    %convolution2dLayer(3,32,'Padding','same','Stride',1, 'Name','conv_3', 'BiasInitializer', @(sz) rand(sz) * 0)
    
    %batchNormalizationLayer('Name', 'norm3')
    
    reluLayer('Name','relu_3')
    
    %dropoutLayer(.25,'Name','dropout_2')

    
    fullyConnectedLayer(15,'Name','fc_1','WeightsInitializer',@(sz) randn(sz) * 0.1,  'BiasInitializer', @(sz) rand(sz) * 0)
    softmaxLayer('Name','softmax')    
    classificationLayer('Name','output', 'Classes',classes)];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)
    
    %factor = 2;
    %layer = setL2Factor(layers,'conv_1/Weights',factor);
    
    %network = dlnetwork(lgraph)
    
    %layers(2) = setL2Factor(layers(2),"Alpha",2);
    %layers(6) = setL2Factor(layers(6),"Alpha",2);
    %layers(10) = setL2Factor(layers(10),"Alpha",2);
    %layers(13) = setL2Factor(layers(13),"Alpha",2);

    
    
    
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsval, ... 
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf,...
    'Verbose',false, ...
    'MiniBatchSize',32, ...'L2Regularization',1e-10, ...'LearnRateSchedule','piecewise', 'LearnRateDropPeriod',2 , 'LearnRateDropFactor',0.7 , ...
    'Plots','training-progress');



%%
net = trainNetwork(augimds,layers,options);


%% Assessment
TestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(TestDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(imread(x))/divideby;

%imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);
augimdsTest = augmentedImageDatastore(imageSize,imdsTest);

% apply the network to the test set
YPredicted = classify(net,augimdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)


%% Assessment LOOP

for i = 1:15
    nets(i) = trainNetwork(augimds,layers,options);
end

%%
TestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(TestDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(imread(x))/divideby;

augimdsTest = augmentedImageDatastore(imageSize,imdsTest);
%% Ensambles of network
goodNets= nets([3, 5, 8, 10, 11])
% apply the network to the test set
for i = 1:5
    net = goodNets(i)
    predictions(:,i) = classify(net,augimdsTest);
          
end

for i = 1 : length(predictions(:,1))
    avaragePrediction(i) = majorityvote(predictions(i,:))

end

avaragePrediction = avaragePrediction'


YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(avaragePrediction == YTest)/numel(YTest)

% confusion matrix
%%
figure
plotconfusion(YTest,avaragePrediction)





