dataDir = 'D:\matlab_jiedan\SR\train';
trainImagesDir = fullfile(dataDir);
exts = [".jpg",".bmp",".png"];
pristineImages = imageDatastore(trainImagesDir,FileExtensions=exts);

%%准备训练数据
upsampledDirName = trainImagesDir+"upsampledImages";
residualDirName = trainImagesDir+"residualImages";
scaleFactors = [2 3 4 8];
createVDSRTrainingSet(pristineImages,scaleFactors,upsampledDirName,residualDirName);