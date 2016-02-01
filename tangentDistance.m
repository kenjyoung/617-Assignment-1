function [ dist ] = tangentDistance( image1 , tangents1, image2, tangents2)
%TANGENTDISTANCE compute the tangent distance between two input images
assert(size(image1) == size(image2), 'image sizes must be equal to compute tangent distance');
c1 = size(image1,1)/2
c2 = size(image1,2)/2
[X,Y] = meshgrid(-c1:size(image1,1)-c1,-c2:size(image1,2)-c2)
[dx1, dy1] = imagegradientxy(image1);
[dx2, dy2] = imagegradientxy(image2);

drot1 = [dx1,dy1].*[Y,-X]
drot2 = [dx2,dy2].*[Y,-X]

dscale1 = [dx1,dy1].*[X,Y]
dscale2 = [dx2,dy2].*[X,Y]

daspect1 = [dx1, dy1].*[X,-Y]
daspect2 = [dx2, dy2].*[X,-Y]

dshear1 = [dx1, dy1].*[Y,0]
dshear2 = [dx2, dy2].*[Y,0]








end

