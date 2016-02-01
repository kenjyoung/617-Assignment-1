function [ dist ] = tangentDistance( image1 , M1, image2, M2)
%TANGENTDISTANCE compute the tangent distance between two input images
assert(isequal(size(image1),size(image2)), 'image sizes must be equal to compute tangent distance');
I1 = image1(:);
I2 = image2(:);

size([M1'*M1,-M1'*M2; M2'*M1,-M2'*M2])
size([M1'*(I2-I1); M2'*(I2-I1)])

u = [M1'*M1,-M1'*M2; M2'*M1,-M2'*M2]\[M1'*(I2-I1); M2'*(I2-I1)];
u1 = u(1:size(u)/2);
u2 = u(size(u)/2+1:end);

dist = norm(I1 +M1*u1 - I2 +M2*u2);

end

