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

k = data_block_dct./quantization_table+0.5;

% What do we gain by encoding/quantizing the image in the transform domain 
% rather than encoding/quantizing the original image pixels directly?

% Can essentially use one value for 8 blocks (the one in the upper left corner)
% The transformed blockgives us the mean of all the values in the block
% (upper left corner) and the variations in the block (the rest of the
% values) 


%% c) reconstruction
%Spør om denne!!!
% dequantizing the values? How? 
de_quant = k.*quantization_table;

reconstructed_block = idct2(de_quant);

psnr_reconstructed_block = "some other code here";

%% Task 2 - Prediction
% Skal lage en "optimal linear predictor" for x(n)
% $$R_x(0) = \sigma_x, R_x(1) = a, R_x(k) = 0 for k>1$$
% 


%Second order predictor:
% $$m(x) = a*x^2 + b*x +c$$




%% Ekstra kode

% J = imresize(data_block, 100);
% 
% figure(1)
% % set(4, "Position",[500,500,8,8])
% % set(gca, "Position", [0,0,1,1])
% imshow(J, [])
% title("8x8 block")


% Koden under funker på samme måte som linjen over hihi
% K = zeros(8,8);
% 
% for i=1:8
%     for j=1:8
%         val = data_block_dct(i,j)/quantization_table(i,j) + 0.5;
%         K(i,j) = val;
%     end
% end








