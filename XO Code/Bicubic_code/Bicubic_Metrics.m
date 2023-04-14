function Bicubic_Metrics(net,testImages,scaleFactors)
% Output_path='D:\matlab_jiedan\SR\save-Out-images\bicubic-high-resolution\';
tic;
for scaleFactor = scaleFactors
    fprintf('Results for Scale factor %d\n\n',scaleFactor);
    
    for idx = 1:numel(testImages.Files)
%         disp(['测试的图像数为：',num2str(numel(testImages.Files))])
        t=clock;
        Img = readimage(testImages,idx);
        img_Iycbcr = rgb2ycbcr(Img);
        img_Ireference = im2double(img_Iycbcr);
        
        % Resize the reference image by a scale factor of 4 to create a low-resolution image using bicubic interpolation
        lowResolutionImg = imresize(img_Ireference,1/scaleFactor,'bicubic');
        
        % Upsample the low-resolution image using bicubic interpolation
        upsampledImg = imresize(lowResolutionImg,[size(img_Ireference,1) size(img_Ireference,2)],'bicubic');
        
        % Separate the upsampled low-resolution image into luminance and color components.
        img_Iy  = upsampledImg(:,:,1);
        img_Icb = upsampledImg(:,:,2);
        img_Icr = upsampledImg(:,:,3);
        
        % Recreate the upsampled image from the luminance and color components and
        % convert to RGB colorspace
        img_Ibicubic = ycbcr2rgb(cat(3,img_Iy,img_Icb,img_Icr));

%         imwrite(img_Ibicubic,[Output_path,int2str(scaleFactor),int2str(idx),'.jpg']);
        
        % Pass the luminance component of upsampled low-resolution image through
        % the trained network and observe the activations from the last layer i.e.
        % Regression Layer. The output of the network is the desired residual
        % image. We pass only the luminance component through the network because
        % we used only the luminance channel while training. The color components
        % are upsampled using bicubic interpolation.
        residualImg = activations(net,img_Iy,'FinalRegressionLayer');
        residualImg = double(residualImg);
        
        % Add the residual image from the network to the upsampled luminance
        % component to get the high-resolution network output
        img_Isr = img_Iy + residualImg;
        
        % Concatenate the upsampled luminance and color components and convert to RGB colorspace to get the final
        % high-resolution color image
        img_Ivdsr = ycbcr2rgb(cat(3,img_Isr,img_Icb,img_Icr));
        
        % Convert the reference image to RGB colorspace
        img_Ireference = ycbcr2rgb(img_Ireference);
        
        % Compare the PSNR and SSIM of the super-resolved image using bicubic interpolation versus that using VDSR. The difference in
        % PSNR shows how much the network improved the image resolution. Higher
        % PSNR and SSIM values generally indicate better results.
        
        % PSNR
        img_bicubicPSNR(idx) = psnr(img_Ibicubic,img_Ireference); %#ok<*AGROW>
%         vdsrPSNR(idx) = psnr(Ivdsr,Ireference);
        
        
        % SSIM
        img_bicubicSSIM(idx) = ssim(img_Ibicubic,img_Ireference);
%         vdsrSSIM(idx) = ssim(Ivdsr,Ireference);

       
        % NIQE
        img_bicubicNIQE(idx) = niqe(img_Ibicubic);
%         vdsrNIQE(idx) = niqe(Ivdsr);

        % SNR
        img_bicubicSNR(idx) = SNR(img_Ireference,img_Ibicubic); 
%         vdsrSNR(idx) = SNR(Ireference,Ivdsr);
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
    img_avgBicubicT = mean(T);
    img_stdBicubicT = std(T,1);
    img_avgBicubicPSNR = mean(img_bicubicPSNR);
    img_stdBicubicPSNR = std(img_bicubicPSNR,1);
    img_avgBicubicNIQE = mean(img_bicubicNIQE);
    img_stdBicubicNIQE = std(img_bicubicNIQE,1);
    img_avgBicubicSNR = mean(img_bicubicSNR);
    img_stdBicubicSNR = std(img_bicubicSNR,1);

    fprintf('Average Time for Bicubic = %f\n',img_avgBicubicT);
    fprintf('Standard dev Time for Bicubic = %f\n',img_stdBicubicT);
    fprintf('Average PSNR for Bicubic = %f\n',img_avgBicubicPSNR);
    fprintf('Standard dev PSNR for Bicubic = %f\n',img_stdBicubicPSNR);
    fprintf('Average SNR for Bicubic = %f\n',img_avgBicubicSNR);
    fprintf('Standard dev SNR for Bicubic = %f\n',img_stdBicubicSNR);
    fprintf('Average NIQE for Bicubic = %f\n',img_avgBicubicNIQE);
    fprintf('Standard dev NIQE for Bicubic = %f\n\n',img_stdBicubicNIQE);


    % Average SSIM for each test set
    % Bicubic
    img_avgBicubicSSIM = mean(img_bicubicSSIM);
    img_stdBicubicSSIM = std(img_bicubicSSIM,1);
    fprintf('Average SSIM for Bicubic = %f\n',img_avgBicubicSSIM);
    fprintf('Standard dev SSIM for Bicubic = %f\n\n',img_stdBicubicSSIM);

    
end
toc;
disp(['Time for Bicubic : ',num2str(toc)]);
end