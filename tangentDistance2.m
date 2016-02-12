function [ tangentDist ] = tangentDistance2( v1, data )
%TANGENTDISTANCE2 Summary of this function goes here
%   Detailed explanation goes here
image_size = [28,28];
num_tangent_vectors = 6;
for i=1:size(data,1)
    image1 = reshape(v1(1:image_size(1)*image_size(2)),image_size(1),image_size(2));
    M1 = reshape(v1(image_size(1)*image_size(2)+1:end),[],num_tangent_vectors);
    image2 = reshape(data(i,1:image_size(1)*image_size(2)),image_size(1),image_size(2));
    M2 = reshape(data(i,image_size(1)*image_size(2)+1:end),[],num_tangent_vectors);
    tangentDist(i) = tangentDistance( image1 , M1, image2, M2);
end
tangentDist = tangentDist';
end

