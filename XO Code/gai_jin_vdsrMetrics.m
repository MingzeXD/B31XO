function gai_jin_vdsrMetrics(net,testImages,scaleFactors)
% Output_path1='D:\matlab_jiedan\SR\save-Out-images\Low-Resolution\';
% Output_path='D:\matlab_jiedan\SR\save-Out-images\test_net_high-resolution\';

tic;
for scaleFactor = scaleFactors
    fprintf('Results for Scale factor %d\n\n',scaleFactor);
    
    for idx = 1:numel(testImages.Files)
%         disp(['测试的图像数为：',num2str(numel(testImages.Files))])
        t=clock;
        Img = readimage(testImages,idx);
        Iycbcr_img = rgb2ycbcr(Img);
        Ireference_img = im2double(Iycbcr_img);
        
        % Resize the reference image by a scale factor of 4 to create a low-resolution image using bicubic interpolation
        lowResolutionImg = imresize(Ireference_img,1/scaleFactor,'bicubic');
        
        % Upsample the low-resolution image using bicubic interpolation
        upsampledImg = imresize(lowResolutionImg,[size(Ireference_img,1) size(Ireference_img,2)],'bicubic');
        
        % Separate the upsampled low-resolution image into luminance and color components.
        Iy_img  = upsampledImg(:,:,1);
        Icb_img = upsampledImg(:,:,2);
        Icr_img = upsampledImg(:,:,3);
        
        LR_Iy_img  = lowResolutionImg(:,:,1);
        LR_Icb_img = lowResolutionImg(:,:,2);
        LR_Icr_img = lowResolutionImg(:,:,3);
        % Recreate the upsampled image from the luminance and color components and
        % convert to RGB colorspace
        Ibicubic_img = ycbcr2rgb(cat(3,Iy_img,Icb_img,Icr_img));
        LR_lowResolutionImg = ycbcr2rgb(cat(3,LR_Iy_img,LR_Icb_img,LR_Icr_img));

%         imwrite(LR_lowResolutionImg,[Output_path1,int2str(scaleFactor),int2str(idx),'.jpg']);
        
        % Pass the luminance component of upsampled low-resolution image through
        % the trained network and observe the activations from the last layer i.e.
        % Regression Layer. The output of the network is the desired residual
        % image. We pass only the luminance component through the network because
        % we used only the luminance channel while training. The color components
        % are upsampled using bicubic interpolation.
        residualImg = activations(net,Iy_img,'FinalRegressionLayer');
        residualImg = double(residualImg);
        
        % Add the residual image from the network to the upsampled luminance
        % component to get the high-resolution network output
        Isr_img = Iy_img + residualImg;
        
        % Concatenate the upsampled luminance and color components and convert to RGB colorspace to get the final
        % high-resolution color image
        Ivdsr_img = ycbcr2rgb(cat(3,Isr_img,Icb_img,Icr_img));
%         imwrite(Ivdsr_img,[Output_path,int2str(scaleFactor),int2str(idx),'.jpg']);

        % Convert the reference image to RGB colorspace
        Ireference_img = ycbcr2rgb(Ireference_img);
        
        % Compare the PSNR and SSIM of the super-resolved image using bicubic interpolation versus that using VDSR. The difference in
        % PSNR shows how much the network improved the image resolution. Higher
        % PSNR and SSIM values generally indicate better results.
        
        % PSNR
%         img_bicubicPSNR(idx) = psnr(Ibicubic_img,Ireference_img);
        img_test_net_PSNR(idx) = psnr(Ivdsr_img,Ireference_img);
        
        
        % SSIM
%         img_bicubicSSIM(idx) = ssim(Ibicubic_img,Ireference_img);
        img_test_net_vdsrSSIM(idx) = ssim(Ivdsr_img,Ireference_img);

       
        % NIQE
%         img_bicubicNIQE(idx) = niqe(Ibicubic_img);
        img_test_net_vdsrNIQE(idx) = niqe(Ivdsr_img);

        % SNR

%         img_bicubicSNR(idx) = SNR(Ireference_img,Ibicubic_img); 
        img_test_net_vdsrSNR(idx) = SNR(Ireference_img,Ivdsr_img);

        T(idx)= etime(clock,t);
        
    end
    
    % Average PSNR for each test set
    % Bicubic
%     tic;
%     toc
    %disp(['运行时间: ',num2str(toc)]);
% 
%     t=clock;
%     etime(clock,t)
    avgBicubicT = mean(T);
    stdBicubicT = std(T,1);
%     img_avgBicubicPSNR = mean(img_bicubicPSNR);
%     img_stdBicubicPSNR = std(img_bicubicPSNR,1);
%     img_avgBicubicNIQE = mean(img_bicubicNIQE);
%     img_stdBicubicNIQE = std(img_bicubicNIQE,1);
%     img_avgBicubicSNR = mean(img_bicubicSNR);
%     img_stdBicubicSNR = std(img_bicubicSNR,1);
% 
    fprintf('Average Time for test-net = %f\n',avgBicubicT);
    fprintf('Standard dev Time for test-net = %f\n',stdBicubicT);
%     fprintf('Average PSNR for Bicubic = %f\n',img_avgBicubicPSNR);
%     fprintf('Standard dev PSNR for Bicubic = %f\n',img_stdBicubicPSNR);
%     fprintf('Average SNR for Bicubic = %f\n',img_avgBicubicSNR);
%     fprintf('Standard dev SNR for Bicubic = %f\n',img_stdBicubicSNR);
%     fprintf('Average NIQE for Bicubic = %f\n',img_avgBicubicNIQE);
%     fprintf('Standard dev NIQE for Bicubic = %f\n\n',img_stdBicubicNIQE);


    % test-net
    img_avgVdsrPSNR = mean(img_test_net_PSNR);
    img_stdVdsrPSNR = std(img_test_net_PSNR,1);
    img_avgVdsrNIQE = mean(img_test_net_vdsrNIQE);
    img_stdVdsrNIQE = std(img_test_net_vdsrNIQE,1);
    img_avgVdsrSNR = mean(img_test_net_vdsrSNR);
    img_stdVdsrSNR = std(img_test_net_vdsrSNR,1);


    fprintf('Average PSNR for test-net = %f\n',img_avgVdsrPSNR);
    fprintf('Standard dev PSNR for test-net = %f\n',img_stdVdsrPSNR);
    fprintf('Average SNR for test-net = %f\n',img_avgVdsrSNR);
    fprintf('Standard dev SNR for test-net = %f\n',img_stdVdsrSNR);
    fprintf('Average NIQE for test-net = %f\n',img_avgVdsrNIQE);
    fprintf('Standard dev NIQE for test-net = %f\n\n',img_stdVdsrNIQE);

    % Average SSIM for each test set
    % Bicubic
%     img_avgBicubicSSIM = mean(img_bicubicSSIM);
%     img_stdBicubicSSIM = std(img_bicubicSSIM,1);
%     fprintf('Average SSIM for Bicubic = %f\n',img_avgBicubicSSIM);
%     fprintf('Standard dev SSIM for Bicubic = %f\n\n',img_stdBicubicSSIM);

    % test-net
    img_avgVdsrSSIM = mean(img_test_net_vdsrSSIM);
    img_stdVdsrSSIM = std(img_test_net_vdsrSSIM,1);
    fprintf('Average SSIM for test-net = %f\n',img_avgVdsrSSIM);
    fprintf('Standard dev SSIM for test-net = %f\n\n',img_stdVdsrSSIM);


    
end
toc;
disp(['test-net ',num2str(toc)]);
end