function [ varargout ] = RESIZE( Size, varargin )

mask = varargin{1};

[block1, block2] = find(mask == 1);
block1 = min(block1):max(block1);
block2 = min(block2):max(block2);

for i = 1:nargin-1
    temp = varargin{i};
    varargin{i} = temp(block1, block2, :);
end
% 
for i = 1:nargin-1
    temp = varargin{i};
    varargin{i} = imresize(temp, Size);
end
%  
% block1 = 176:215;
% block2 = 146:185;
% 
% % block1 = 61:338;
% % block2 = 170:735;
% specImg = specImg(block1, block2, :);
% mask = mask(block1, block2, :);
% gt_shad = gt_shad(block1, block2, :);
% gt_refl = gt_refl(block1, block2, :);
for i = 1:nargin-1
    varargout{i} = varargin{i} .* repmat(varargin{1}, [1, 1, size(varargin{i}, 3)]);
end
