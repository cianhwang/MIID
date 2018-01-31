function [ Wr, Wd ] = weightMap( X , beta, flag)

[row col band] = size(X);

if nargin < 2
    alpha =50000;
    beta = 0.0003;
end
%     beta = 0.0005;

%% right
right = X(:, [2:end, 1], :);

x = zeros(band,1);
x2  = zeros(band, 1);
Wr = zeros(row, col);
for m = 1:row
    for n = 1:col
        x(1:band) = X(m, n, :);
        x2(1:band) = right(m, n, :);
        
        if norm(x)*norm(x2) == 0
            if norm(x)==0 && norm(x2)==0
                w = 1;
            else
                w = 0;
            end
        else
            w = x'*x2/(norm(x)*norm(x2));
        end
        Wr(m, n) =1-w;
    end
end
Wr = im2bw(Wr, beta);
% Wr = 1./(1+exp(-alpha*(Wr - beta)));
if flag ==1
Wr = bwmorph(Wr,'skel',inf);
end
Wr = mapDenoise(Wr);
%-----------------------------
% MapMap = double(edge(rgb2gray(X), 'Canny'));
% MapMap = MapMap(:, [2:end, 1]) - MapMap;
% Wr = Wr.*MapMap;
figure, imshow(Wr);

%% down
down = X([2:end, 1], :, :);

x = zeros(band,1);
x2  = zeros(band, 1);
Wd = zeros(row, col);
for m = 1:row
    for n = 1:col
        x(1:band) = X(m, n, :);
        x2(1:band) = down(m, n, :);
        
        if norm(x)*norm(x2) == 0
            if norm(x)==0 && norm(x2)==0
                w = 1;
            else
                w = 0;
            end
        else
            w = x'*x2/(norm(x)*norm(x2));
        end

        Wd(m, n) =  1-w;
    end
end
Wd = im2bw(Wd, beta);

% Wd = 1./(1+exp(-alpha*(Wd - beta)));

if flag ==1
Wd = bwmorph(Wd,'skel',Inf);
end
Wd = mapDenoise(Wd);
%---------------------------------
% MapMap = double(edge(rgb2gray(X), 'Canny'));
% MapMap = MapMap([2:end, 1], :) - MapMap;
% Wd = Wd .* MapMap;
figure, imshow(Wd);


function W1 = mapDenoise(W)
bwW = im2bw(W, 0.1);
bwW=bwareaopen(bwW,5);

W1 = W.*bwW;
