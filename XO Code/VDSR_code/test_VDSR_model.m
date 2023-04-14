%%%加载模型

load ('VDSR-net\net_checkpoint__55875__2023_03_25__22_07_09.mat') 
%%用同样数据集进行训练VDSR网络，得到的VDSR网络模型，为了与改进后的模型进行对比




%%%测试单个图像的PSNR和SSIM，对比算法为传统经典算法：双三次插值。也可以换成2016年提出的VDSR网络。
%%%对该低分辨率图像执行 SISR，双三次插值是一种不依赖于深度学习的传统图像处理解决方案。
testImg = "cygloop_exp1_log_4_1.fits.jpg";
img_Ireference = imread(testImg);
img_Ireference = im2double(img_Ireference);
% imshow(Ireference)
% title("High-Resolution Reference Image")
scaleFactor = 0.25; %%尺度因子为 1/2， 1/3，1/4，1/8
img_Ilowres = imresize(img_Ireference,scaleFactor,"bicubic");
% imshow(Ilowres)
% title("Low-Resolution Image")
[nrows,ncols,np] = size(img_Ireference);
img_Ibicubic = imresize(img_Ilowres,[nrows ncols],"bicubic");
% figure(1);
% imshow(Ibicubic)
% title("High-Resolution Image Obtained Using Bicubic Interpolation")

img_Iycbcr = rgb2ycbcr(img_Ilowres);
img_Iy = img_Iycbcr(:,:,1);
img_Icb = img_Iycbcr(:,:,2);
img_Icr = img_Iycbcr(:,:,3);

img_Iy_bicubic = imresize(img_Iy,[nrows ncols],"bicubic");
img_Icb_bicubic = imresize(img_Icb,[nrows ncols],"bicubic");
img_Icr_bicubic = imresize(img_Icr,[nrows ncols],"bicubic");

img_Iresidual = activations(net,img_Iy_bicubic,'FinalRegressionLayer');

img_Iresidual = double(img_Iresidual);
% % imshow(Iresidual,[])
% % title("Residual Image from best-net")


img_Isr = img_Iy_bicubic + img_Iresidual;
img_Ivdsr = ycbcr2rgb(cat(3,img_Isr,img_Icb_bicubic,img_Icr_bicubic));
% figure(2);
% imshow(Ivdsr)
% title("High-Resolution Image Obtained Using best-net")

% roi = [360 50 400 350];
% 
% montage({imcrop(Ibicubic,roi),imcrop(Ivdsr,roi)})
% title("High-Resolution Results Using Bicubic Interpolation (Left) vs. VDSR (Right)");

% PSNR
img_bicubicPSNR = psnr(img_Ibicubic,img_Ireference);
VDSR_PSNR = psnr(img_Ivdsr,img_Ireference);
% SSIM
bicubicSSIM = ssim(img_Ibicubic,img_Ireference);
VDSR_SSIM = ssim(img_Ivdsr,img_Ireference);

% 使用自然图像质量评价方法 (NIQE) 测量图像感知质量。NIQE 分数越小，表示感知质量越好
bicubicNIQE = niqe(img_Ibicubic);
VDSR_NIQE = niqe(img_Ivdsr);


figure;
subplot(1,3,1);

imshow(rgb2gray(img_Ireference))
colorbar;
axis on
title("Original image")

subplot(1,3,2);
imshow(rgb2gray(img_Ilowres));
colorbar;
axis on
title("Low-Resolution Image")

subplot(1,3,3);
imshow(rgb2gray(img_Ivdsr));
colorbar;
axis on
title(["High-Resolution Image for VDSR：",'PSNR:',num2str(VDSR_PSNR,'%2.2f'),'dB','SSIM:',num2str(VDSR_SSIM,'%2.4f'),'scaleFactor:',num2str(scaleFactor)])


disp(['Bicubic PSNR:',num2str(img_bicubicPSNR)])
disp(['VDSR PSNR:',num2str(VDSR_PSNR)])

disp(['Bicubic SSIM:',num2str(bicubicSSIM)])
disp(['VDSR SSIM:',num2str(VDSR_SSIM)])

disp(['Bicubic NIQE:',num2str(bicubicNIQE)])
disp(['VDSR NIQE:',num2str(VDSR_NIQE)])

% %批量测试整个数据集的平均psnr ssim
% dataDir = 'C:\Users\dell\Desktop\SR\SR\test';
% testImagesDir = fullfile(dataDir);
% exts = [".jpg",".bmp",".png"];
% pristineImages = imageDatastore(testImagesDir,FileExtensions=exts);
% 
% scaleFactors = [2 3 4 8];
% test_vdsrMetrics(net,pristineImages,scaleFactors);
