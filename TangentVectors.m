function [ vectors ] = TangentVectors( images )
%TANGENTVECTORS Summary of this function goes here
%   Detailed explanation goes here

num_images = size(images,3);

H = fspecial('gaussian',5,0.4);
for i=1:num_images
    image = imfilter(images(:,:,i), H);
    c = size(image)/2;

    [X,Y] = meshgrid(-c(1):size(image,1)-c(1)-1,-c(2):size(image,2)-c(2)-1);
    [dx, dy] = imgradientxy(image);

    drot = dx.*Y-dy.*X;

    dscale = dx.*X+dy.*Y;

    daspect = dx.*X-dy.*Y;

    dshear = dx.*Y;
    
    dthickness = dx*dx+dy*dy;

    vectors(:,:,i) = normc([dx(:),dy(:),drot(:),dscale(:),daspect(:),dshear(:), dthickness(:)]);
    %vectors(:,:,i) = [dx,dy,drot,dscale,daspect,dshear];
end
end

