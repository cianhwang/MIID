function [ Wr, Wd ] = weightMap( input , alpha, beta)

if nargin < 2
    beta = 0.0003;
	alpha =1000;
end

%% right
right = input(:, [2:end, 1], :);

d = 1 - sum(right.*input,3);
d(find(d==1))=0; 
Wr = 1./(1+exp(-alpha*(d - beta)));

Wr = mapDenoise(Wr);

%% down
down = input([2:end, 1], :, :);

d = 1 - sum(down.*input,3);
d(find(d==1))=0;  %#ok<*FNDSB>
Wd = 1./(1+exp(-alpha*(d - beta)));

Wd = mapDenoise(Wd);


function W1 = mapDenoise(W)
bwW = imbinarize(W, 0.5);
bwW=bwareaopen(bwW,15);%Angela: 删除二值图像中的小面积对象

W1 = W.*bwW;