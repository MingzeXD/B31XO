 function snr=SNR(im,im_denoised) 
%      Ps=sum(sum(I.^2)); 
%      Pn=sum(sum((In-I).^2)); 
%      snr=10*log10(Ps/Pn);
     
    signal_energy = sum(im(:).^2);

    % 计算噪声能量
    noise = double(im) - double(im_denoised);
    noise_energy = sum(noise(:).^2);

    % 计算信噪比
    snr = 10 * log10(signal_energy / noise_energy);
