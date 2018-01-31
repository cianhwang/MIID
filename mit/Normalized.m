function [ II ] = Normalized( I )

[m, n, k] = size(I);
map = zeros(m,n);
x = zeros(k, 1);
II = zeros(m, n, k);
for i = 1:m
    for j = 1:n
        x(1:k,1) = I(i, j, :);
        map(i, j) = norm(x);
        if norm(x) ~= 0
        II(i, j, :) = I(i, j, :)./map(i, j);
        end
    end
end
if k == 3
    RGB = II;
else
    RGB = convRGB(II, 410, 700, 290/58);
end
figure, imshow(RGB);
