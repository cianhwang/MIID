function [ V ] = Entropy( grayImg )

grayImg = (grayImg-min(grayImg(:)))/(max(grayImg(:))-min(grayImg(:)));
grayImg = uint8(grayImg*255);

Hist = imhist(grayImg);
p = zeros(255, 1);
N = 30;
for i = 1:255
    for j = max(1, (i-N/2)) : min((i+N/2), 255)
        p(i) = p(i)+gaussmf(i, [j, 1])*Hist(j);
    end
    p(i) = p(i)/N;
end

V = sum(p);


