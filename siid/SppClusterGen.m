function [ spp_new_label ] = SppClusterGen(spp_label,spp_rgb,spp_am,spp_kmeans_k)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
label_size = max(size(spp_label));
spp_kmeans_label = kmeans(spp_rgb,spp_kmeans_k,'distance','cosine');
spp_new_label = zeros(label_size,1);
spp_finish_label = zeros(label_size,1);
for i = 1:label_size
    if spp_finish_label(i) == 1
        continue;
    end
    spp_new_label(i) = i;
    stock_ptr = 1;
    stock = zeros(label_size,1);
    stock(stock_ptr) = i;
    stock_ptr = stock_ptr+1;
    while stock_ptr ~= 1
        stock_ptr = stock_ptr-1;
        label_id = stock(stock_ptr);
        am_label_ids = find(spp_am(label_id,:) == 1);
        am_label_size = max(size(am_label_ids));
        for j = 1:am_label_size
            if spp_finish_label(am_label_ids(j)) == 1
                continue;
            end
            if spp_kmeans_label(i) == spp_kmeans_label(am_label_ids(j))
                spp_new_label(am_label_ids(j)) = i;
                spp_finish_label(am_label_ids(j)) = 1;
                stock(stock_ptr) = am_label_ids(j);
                stock_ptr = stock_ptr+1;
            end
        end
    end
end
end
function a = GetNorm(a)
[~,d] = size(a);
lens = sum(a,2);
a = a./repmat(lens,[1,d]);
end

