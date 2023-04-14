load ('best_model\best_net.mat')
% load ('VDSR-net\net_checkpoint__55875__2023_03_25__22_07_09.mat') 
% %load('net_model\net_checkpoint__44700__2023_03_25__15_08_58.mat')

testImg = "cygloop_exp1_log_4_1.fits.jpg";
img_Ireference = imread(testImg);
img_Ireference = im2double(img_Ireference);
% imshow(Ireference)
% title("High-Resolution Reference Image")
scaleFactor = 0.25; 
img_Ilowres = imresize(img_Ireference,scaleFactor,"bicubic");
% imshow(Ilowres)
% title("Low-Resolution Image")
[nrows,ncols,np] = size(img_Ireference);
img_Ibicubic = imresize(img_Ilowres,[nrows ncols],"bicubic");

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

% roi = [360 50 400 350];
% 
% montage({imcrop(Ibicubic,roi),imcrop(Ivdsr,roi)})
% title("High-Resolution Results Using Bicubic Interpolation (Left) vs. VDSR (Right)");

% PSNR
bicubicPSNR = psnr(img_Ibicubic,img_Ireference);
test_net_PSNR = psnr(img_Ivdsr,img_Ireference);
% SSIM
bicubicSSIM = ssim(img_Ibicubic,img_Ireference);
test_net_SSIM = ssim(img_Ivdsr,img_Ireference);

bicubicNIQE = niqe(img_Ibicubic);
test_net_NIQE = niqe(img_Ivdsr);

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
title(["High-Resolution Image for test-net：",'PSNR:',num2str(test_net_PSNR,'%2.2f'),'dB','SSIM:',num2str(test_net_SSIM,'%2.4f'),'scaleFactor:',num2str(scaleFactor)])

disp(['Bicubic PSNR:',num2str(bicubicPSNR)])
disp(['test-net PSNR:',num2str(test_net_PSNR)])

disp(['Bicubic SSIM:',num2str(bicubicSSIM)])
disp(['test-net SSIM:',num2str(test_net_SSIM)])

disp(['Bicubic NIQE:',num2str(bicubicNIQE)])
disp(['test-net NIQE:',num2str(test_net_NIQE)])

% dataDir = 'C:\Users\dell\Desktop\SR\SR\test';
% testImagesDir = fullfile(dataDir);
% exts = [".jpg",".bmp",".png"];
% pristineImages = imageDatastore(testImagesDir,FileExtensions=exts);
% scaleFactors = [2 3 4 8];
% gai_jin_vdsrMetrics(net,pristineImages,scaleFactors);

