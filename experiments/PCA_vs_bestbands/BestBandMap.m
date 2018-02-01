function [ X ] = BestBandMap(specImg, Jr)

[Row, Col, Chn] = size(specImg);
K = Chn;
N = Row*Col;

% maybe K-L divergence could also be useful here
SSIM = zeros(K, K);

for i = 1: K
    for j = (i+1):K
    SSIM(i, j) = ssim(specImg(:, :, i), specImg(:, :, j));
    end
end

SSIM = SSIM'+SSIM;

xxx = zeros(1, Jr);
for i = 1:Jr
    [M, I] = min(mean(SSIM, 1));
    SSIM(:, I) = ones(K, 1);
    SSIM(I, :) = ones(1, K);
    xxx(i) = I;
end

specImgVec = reshape(specImg, [], K);

B = specImgVec;

A = specImgVec(:, xxx);

X = (A'*A)\((A')*B);
X = X';

