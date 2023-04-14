dataDir = 'D:\matlab_jiedan\SR\train';
trainImgsDir = fullfile(dataDir);
exts = [".jpg",".bmp",".png"];
pristineImg = imageDatastore(trainImgsDir,FileExtensions=exts);

%%Prepare training data
img_upsampledDirName = trainImgsDir+"upsampledImages";
img_residualDirName = trainImgsDir+"residualImages";
scaleFactors = [2 3 4 8];
% createVDSRTrainingSet(pristineImages,scaleFactors,upsampledDirName,residualDirName);
%%Define the preprocessing pipeline for the training set
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
%%生成的数据存储 在纪元的每次迭代时向网络提供小批量数据。预览从数据存储读取的结果
% inputBatch = preview(dsTrain);
% disp(inputBatch)

%%设置网络结构
%%感受野和图像补丁大小为 41 x 41。图像输入层接受具有一个通道的图像，仅使用亮度通道进行训练。

first_conv_Layer = imageInputLayer([41 41 1],Name="InputLayer",Normalization="none");

%%图像输入层后跟一个 2-D 卷积层，其中包含 64 个大小为 3×3 的过滤器。
image_convLayer = convolution2dLayer(3,64,Padding=1, ...
    WeightsInitializer="he",BiasInitializer="zeros",Name="Conv1");
%%指定 ReLU 层。
conv_relLayer = reluLayer(Name="ReLU1");

net_middle_Layers = [image_convLayer conv_relLayer];

%%中间层包含卷积层，滤波器，

%%第一个残差块
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv2");
    conv_relLayer = reluLayer(Name="ReLU2");

    net_middle_Layers = [net_middle_Layers image_convLayer conv_relLayer]; 
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv3");
    net_middle_Layers = [net_middle_Layers image_convLayer]; 


    add1 = additionLayer(2,'Name','add_1');
    conv_relLayer = reluLayer(Name="ReLU3");
    net_middle_Layers = [net_middle_Layers add1 conv_relLayer]; 

%%第二个残差块
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv4");
    conv_relLayer = reluLayer(Name="ReLU4");

    net_middle_Layers = [net_middle_Layers image_convLayer conv_relLayer]; 
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv5");
    net_middle_Layers = [net_middle_Layers image_convLayer];


    add2 = additionLayer(2,'Name','add_2');
    conv_relLayer = reluLayer(Name="ReLU5");
    net_middle_Layers = [net_middle_Layers add2 conv_relLayer]; 
    

%%第三个残差块
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv6");
    conv_relLayer = reluLayer(Name="ReLU6");

    net_middle_Layers = [net_middle_Layers image_convLayer conv_relLayer]; 
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv7");

    net_middle_Layers = [net_middle_Layers image_convLayer]; 

    add3 = additionLayer(2,'Name','add_3');
    conv_relLayer = reluLayer(Name="ReLU7");
    net_middle_Layers = [net_middle_Layers add3 conv_relLayer]; 

%%最后一个    
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv8");
    add4 = additionLayer(2,'Name','add_4');
    net_middle_Layers = [net_middle_Layers image_convLayer add4];
    
    conv_relLayer = reluLayer(Name="ReLU8");
    
    image_convLayer = convolution2dLayer(3,64,Padding=[1 1], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv9");

    net_middle_Layers = [net_middle_Layers conv_relLayer image_convLayer];

    concat1 = depthConcatenationLayer(4,'Name','concat_1');
    net_middle_Layers = [net_middle_Layers concat1]; 
%%降维
    image_convLayer = convolution2dLayer(1,64,Padding=[0 0], ...
        WeightsInitializer="he",BiasInitializer="zeros", ...
        Name="Conv10");
    net_middle_Layers = [net_middle_Layers image_convLayer]; 

%%倒数第二层是一个卷积层，具有大小为 3 x 3 x 64 的单个过滤器，用于重建图像。
  image_convLayer = convolution2dLayer(3,1,Padding=[1 1], ...
    WeightsInitializer="he",BiasInitializer="zeros", ...
    NumChannels=64, Name="Conv11");
%%%   add5 = additionLayer(2,'Name','add_5'); %%残差相加，形成最后输出
%   relLayer = reluLayer(Name="ReLU8");
%%最后一层是回归层，而不是 ReLU 层。回归层计算残差图像和网络预测之间的均方误差。
net_final_Layers = [image_convLayer regressionLayer(Name="FinalRegressionLayer")];

% % net_final_Layers = [image_convLayer add5 regressionLayer(Name="FinalRegressionLayer")]; %% 显示最终的结构

%连接所有层以形成网络结构。
net_layers = [first_conv_Layer net_middle_Layers  net_final_Layers];

% for i = 1:numel(layers)
%     disp(layers(i).Name)
% end

net_lgraph = layerGraph(net_layers);

net_lgraph = connectLayers(net_lgraph,'ReLU1','add_1/in2');
net_lgraph = connectLayers(net_lgraph,'ReLU3','add_2/in2');
net_lgraph = connectLayers(net_lgraph,'ReLU5','add_3/in2');
net_lgraph = connectLayers(net_lgraph,'ReLU1','add_4/in2');
% % net_lgraph = connectLayers(net_lgraph,'InputLayer','add_5/in2');
net_lgraph = connectLayers(net_lgraph, 'Conv3', 'concat_1/in2');
net_lgraph = connectLayers(net_lgraph, 'Conv5', 'concat_1/in3');
net_lgraph = connectLayers(net_lgraph, 'Conv7', 'concat_1/in4');

%%显示网络的结构
% figure
% plot(net_lgraph)



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
    CheckpointPath='D:\matlab_jiedan\SR\net_model', ...
    CheckpointFrequency=5,...
    Verbose=false);

%%训练网络

doTraining = true;
if doTraining
    net = trainNetwork(dsTrain,net_lgraph,options);
    createmodelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trained-"+createmodelDateTime+".mat","net");
% else
%     load("trainedNet.mat");
end

