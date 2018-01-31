function [ Wr, Wd ] = weightMap( X , alpha, beta, flag)

[row col band] = size(X);

if nargin < 2
    beta = 0.0003;
     alpha =1000;
end



%% right
right = X(:, [2:end, 1], :);

x = zeros(band,1);
x2  = zeros(band, 1);
Wr = zeros(row, col);
for m = 1:row
    for n = 1:col
        x(1:band) = X(m, n, :);
        x2(1:band) = right(m, n, :);
        
        if norm(x)*norm(x2) < 0.5
%             if norm(x)==0 && norm(x2)==0
                w = 1;
%             else
%                 w = 0;
%             end
        else
            w = x'*x2/(norm(x)*norm(x2));
        end
        Wr(m, n) =1-w;
    end
end
% Wr = im2bw(Wr, beta);
Wr = 1./(1+exp(-alpha*(Wr - beta)));

Wr = mapDenoise(Wr, flag);
% Wr = edge(mean(X,3), 'Canny');
% Wr = Wr - Wr(:, [2:end, 1]);
% figure, imshow(Wr);

%% down
down = X([2:end, 1], :, :);

x = zeros(band,1);
x2  = zeros(band, 1);
Wd = zeros(row, col);
for m = 1:row
    for n = 1:col
        x(1:band) = X(m, n, :);
        x2(1:band) = down(m, n, :);
        
        if norm(x)*norm(x2) < 0.5
%             if norm(x)==0 && norm(x2)==0
                w = 1;
%             else
%                 w = 0;
%             end
        else
            w = x'*x2/(norm(x)*norm(x2));
        end

        Wd(m, n) =  1-w;
    end
end
% Wd = im2bw(Wd, beta);
% 
Wd = 1./(1+exp(-alpha*(Wd - beta)));

Wd = mapDenoise(Wd, flag);
% Wd = edge(mean(X,3), 'Canny');
% Wd = Wd-Wd([2:end, 1],:);
% figure, imshow(Wd);


function W1 = mapDenoise(W, flag)
bwW = im2bw(W, 0.5);
bwW=bwareaopen(bwW,15);
if flag ==1
bwW = bwmorph(bwW,'skel',Inf);
end

W1 = W.*bwW;
