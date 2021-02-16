
%%%%%%%%%%%%%% Waleed Azam (C) 2020
%%%%%%%%%%%%%% Assignment 1, Ex 1, a, b, c;
image= [124, 125, 122, 120, 122, 119, 117, 118;
    121, 121, 120, 119, 119, 120, 120, 118;
    126, 124, 123, 122, 121, 121, 120, 120;
    124, 124, 125, 125, 126, 125, 124, 124;
    127, 127, 128, 129, 130, 128, 127, 125;
    143, 142, 143, 142, 140, 139, 139, 139;
    150, 148, 152, 152, 152, 152, 150, 151;
    156, 159, 158, 155, 158, 158, 157, 156]
%%
Q = [16 11 10 16 24 40 51 61;
    12 12 14 19 26 58 60 55;
    14 13 16 24 40 57 69 56;
    14 17 22 29 51 87 80 62;
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92;
    49 64 78 87 103 121 120 101;
    72 92 95 98 112 100 103 99];
%%
imageDCT = dct2(image);
%%
coeffImage = zeros(size(image, 1), size(image, 2));
coeffTransform = zeros(zeros(image, 1), size(image, 2));
%%
for i = 1: size(image, 1)
    for j= 1:size(image, 2)
        coeffImage(i,j) = floor (image(i,j)/q(i,j) + 0.5);
        coeffTransform(i,j) = floor (imageDCT(i,j)/q(i,j) + 0.5);
    end
end
%%
%Dequantization 
reconstImage= zeros(size(image, 1), size(image, 2));
deQTransform= zeros(size(image, 1), size(image, 2));
for i = 1:size(image, 1)
    for j = 1:size(images, 2)
        reconstImage(i,j)= coeffImage(i,j)*q(i,j) -0.5;
        deQTransform(i,j) = q(i,j)*coeffTransform(i,j) - 0.5;
    end
end
%%
% RECONSTRUCTING AND PLOTTING
reconstTransform= idct2(deQTransform);
psnrImage = psnr(reconstImage, image, 255);
psnrTransform = psnr(reconstTransform, image, 255);
figure(1);
subplot(1,3,1);
imshow(mat2gray(reconstImage));
title('Rekonstruert med berre koffisientar');
subplot(1,3,2);
imshow(mat2gray(image));
title('Originalbildet');
subplot(1,3,3);
imshow(mat2gray(reconstTransform));
title('Rekonstruert basert på DCT');
