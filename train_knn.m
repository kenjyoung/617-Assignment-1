%load mnist data
display 'loading data...'
images = loadMNISTImages('data/train-images-idx3-ubyte');
images = images(:,:,1:1000);
labels = loadMNISTLabels('data/train-labels-idx1-ubyte');
num_images = size(images,3);
fold_size = num_images/10;

%compute tangent vectors
display 'computing tangent vectors'
tangentVectors = TangentVectors(images);
tangentVectors = reshape(tangentVectors, size(tangentVectors,1)*size(tangentVectors,2),...
    size(tangentVectors,3)); 

images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

imageTangentVectors=[images(:,:);tangentVectors(:,:)];
tangentDistance2(imageTangentVectors(:,3)',imageTangentVectors(:,1:end)')
IDX = knnsearch(imageTangentVectors(:,2:end)',imageTangentVectors(:,1)', 'K', 10,...
    'Distance', @tangentDistance2);

labels(3)
labels(IDX)

%print 'preform cross validation'
%for k=1:20
%   for i=1:10
%       
%   end
%end
