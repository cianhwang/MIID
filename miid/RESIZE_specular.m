function [ varargout ] = RESIZE( Size, varargin )

mask = varargin{1};

%% 剪裁到包围目标的最小方框
[row, col] = find(mask == 1);
row = min(row):max(row);
col = min(col):max(col);

for i = 1:nargin-1
    temp = varargin{i};
    varargin{i} = temp(row, col, :);
end

%% 缩放
Size = min(Size/(max(length(row),length(col))),1);
for i = 1:nargin-1
    temp = varargin{i};
    varargin{i} = imresize(temp, Size);
end

%% 乘上mask
for i = 1:nargin-1
    varargout{i} = varargin{i} .* repmat(varargin{1}, [1, 1, size(varargin{i}, 3)]); %#ok<AGROW>
end
