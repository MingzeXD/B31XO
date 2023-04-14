dataDir = 'D:\matlab_jiedan\SR\train';
trainImgDir = fullfile(dataDir);
exts = [".jpg",".bmp",".png"];
pristineImg = imageDatastore(trainImgDir,FileExtensions=exts);

%%准备训练数据
img_upsampledDirName = trainImgDir+"upsampledImages";
img_residualDirName = trainImgDir+"residualImages";
scaleFactors = [2 3 4 8];
% createVDSRTrainingSet(pristineImages,scaleFactors,upsampledDirName,residualDirName);
%%定义训练集的预处理流程
upsampledImg = imageDatastore(img_upsampledDirName,FileExtensions=".mat",ReadFcn=@matRead);
residualImg = imageDatastore(img_residualDirName,FileExtensions=".mat",ReadFcn=@matRead);


%%创建一个 imageDataAugmenter，用于指定数据增强的参数。
% 在训练过程中使用数据增强来改变训练数据，从而有效地增加可用训练数据量。
% 在这里，增强器指定随机旋转 90 度和 x 方向的随机反射。

image_augmenter = imageDataAugmenter( ...
    RandRotatio=@()randi([0,1],1)*90, ...
    RandXReflection=true);
%%创建一个随机补丁提取数据存储，该数据存储从上采样和残差图像数据存储中执行随机补丁提取。
patchSize = [41 41];
patchesPerImage = 64;
dsTrain = randomPatchExtractionDatastore(upsampledImg,residualImg,patchSize, ...
    DataAugmentation=image_augmenter,PatchesPerImage=patchesPerImage);


networkDepth = 20;
firstLayer = imageInputLayer([41 41 1],Name="InputLayer",Normalization="none");

convLayer = convolution2dLayer(3,64,Padding=1, ...
    WeightsInitializer="he",BiasInitializer="zeros",Name="Conv1");
relLayer = reluLayer(Name="ReLU1");


middleLayers = [convLayer relLayer];
for layerNumber = 2:networkDepth-1
    convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv"+num2str(layerNumber));
    
    relLayer = reluLayer(Name="ReLU"+num2str(layerNumber));
    middleLayers = [middleLayers convLayer relLayer];    
end

convLayer = convolution2dLayer(3,1,Padding=[1 1], ...
    WeightsInitializer="he",BiasInitializer="zeros", ...
    NumChannels=64,Name="Conv"+num2str(networkDepth));

% % add1 = additionLayer(2,'Name','add_1'); %%残差相加，形成最后输出

finalLayers = [convLayer  regressionLayer(Name="FinalRegressionLayer")];

% % finalLayers = [convLayer  add1 regressionLayer(Name="FinalRegressionLayer")];
net_layers = [firstLayer middleLayers finalLayers];


net_layers = layerGraph(net_layers);
% layers = connectLayers(layers,'InputLayer','add_1/in2');

%%显示网络的结构
% figure
% plot(layers)


%%指定训练选项
%%使用随机梯度下降和动量 （SGDM） 优化来训练网络。

maxEpochs = 90;
epochIntervals = 1;
initLearningRate = 0.1;
learningRateFactor = 0.1;
l2reg = 0.0001;
miniBatchSize = 64;
options = trainingOptions("sgdm", ...
    Momentum=0.9, ...
    InitialLearnRate=initLearningRate, ...
    ValidationFrequency=50,...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=10, ...
    LearnRateDropFactor=learningRateFactor, ...
    L2Regularization=l2reg, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    GradientThresholdMethod="l2norm", ...
    GradientThreshold=0.01, ...
    Plots="training-progress", ...
    CheckpointPath='D:\matlab_jiedan\SR\VDSR-net', ...
    CheckpointFrequency=5,...
    Verbose=false);

%%训练网络

doTraining = true;
if doTraining
    net = trainNetwork(dsTrain,net_layers,options);
    createmodelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trained-"+createmodelDateTime+".mat","net");
% else
%     load("trainedNet.mat");
end


