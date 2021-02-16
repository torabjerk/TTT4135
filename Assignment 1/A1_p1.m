%% Task 1 - still image compression
data_block = [124 125 122 120 122 119 117 118; 
    121 121 120 119 119 120 120 118;
    126 124 123 122 121 121 120 120;
    124 124 125 125 126 125 124 124;
    127 127 128 129 130 128 127 125;
    143 142 143 142 140 139 139 139;
    150 148 152 152 152 152 150 151;
    156 159 158 155 158 158 157 156];

quantization_table = [16 11 10 16 24 40 51 61;
    12 12 14 19 26 58 60 55;
    14 13 16 24 40 57 69 56;
    14 17 22 29 51 87 80 62;
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92;
    49 64 78 87 103 121 120 101;
    72 92 95 98 112 100 103 99;];

%% a) DCT

data_block_dct = dct2(data_block); % Transform of 8x8 block to dct

%% b) transform coefficients

q_img = floor(data_block./quantization_table+0.5);

k = floor(data_block_dct./quantization_table+0.5);

%% c) reconstruction

de_quant_img = q_img.*quantization_table - 0.5;

de_quant_DCT = k.*quantization_table - 0.5;

reconstructed_block = idct2(de_quant_DCT);


psnr_img_q = psnr(de_quant_img, data_block, 255);
psnr_dct_transform = psnr(reconstructed_block, data_block, 255);

%%
figure(1);
subplot(3,1,1);
imshow(mat2gray(de_quant_img));
title('Reconstructed image using only quantization');
subplot(3,1,2);
imshow(mat2gray(data_block));
title('Original image');
subplot(3,1,3);
imshow(mat2gray(reconstructed_block));
title('Reconstructed using DCT and quantization')








