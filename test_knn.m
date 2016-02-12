clear
display 'loading trained model'
load('train_knn_results')
display 'loading test data'
images = loadMNISTImages('data/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('data/t10k-labels-idx1-ubyte');
trainLabels = loadMNISTLabels('data/train-labels-idx1-ubyte');
num_test_images = size(images,3);

display 'computing tangent vectors'
tangentVectors = TangentVectors(images);
tangentVectors = reshape(tangentVectors, size(tangentVectors,1)*size(tangentVectors,2),...
    size(tangentVectors,3)); 

images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

testImageTangentVectors=[images(:,:);tangentVectors(:,:)];
[IDX, D] = knnsearch(trainImageTangentVectors(:,:)',testImageTangentVectors(:,:)', 'K', k_best,...
        'Distance', @tangentDistance2);
y = mode(trainLabels(IDX),2);
accuracy = nnz(y==testLabels)/num_test_images;
fprintf('Test set accuracy: %f\n',accuracy);