function a = GetL2Norm(a)
[~,d] = size(a);
lens = sum(a.^2,2).^0.5;
a = a./repmat(lens,[1,d]);
end