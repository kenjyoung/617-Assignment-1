%load mnist data
print 'loading data...'
images = loadMNISTImages('data/train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
num_images = size(images,3);
fold_size = num_images/10;

%compute tangent vectors
print 'computing tangent vectors'
tangentVectors = TangentVectors(images);

print 'preform cross validation'
for k=1:20
   for i=1:10
       
   end
end
