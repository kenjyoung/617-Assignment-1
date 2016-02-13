function [ tangentDist ] = tangentDistance2( v1, data )
%TANGENTDISTANCE2 Summary of this function goes here
%   Detailed explanation goes here
image_size = [28,28];
num_tangent_vectors = 7;
k_filter = 100;
image1 = v1(1:image_size(1)*image_size(2));
M1 = reshape(v1(image_size(1)*image_size(2)+1:end),[],num_tangent_vectors);
if(size(data,1)>k_filter) 
    filtered_set = knnsearch(data(:,1:image_size(1)*image_size(2)),image1,...
        'K', k_filter, 'Distance', 'euclidean');
else
    filtered_set = 1:size(data,1);
end
tangentDist= Inf(1,size(data,1));
for i=filtered_set
    image2 = data(i,1:image_size(1)*image_size(2));
    M2 = reshape(data(i,image_size(1)*image_size(2)+1:end),[],num_tangent_vectors);
    tangentDist(:,i) = tangentDistance( image1 , M1, image2, M2);
end
tangentDist = tangentDist';
end

