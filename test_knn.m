clear
display 'loading trained model'
load('train_knn_results')
display 'loading test data'
testImages = loadMNISTImages('data/t10k-images-idx3-ubyte');
trainImages = loadMNISTImages('data/train-images-idx3-ubyte');
testLabels = loadMNISTLabels('data/t10k-labels-idx1-ubyte');
trainLabels = loadMNISTLabels('data/train-labels-idx1-ubyte');
num_test_images = size(testImages,3);

display 'computing tangent vectors'
trainImageTangentVectors = TangentVectors(trainImages);
trainImageTangentVectors = reshape(trainImageTangentVectors, ...
    size(trainImageTangentVectors,1)*size(tangentVectors,2),...
    size(trainImageTangentVectors,3)); 
tangentVectors = TangentVectors(testImages);
tangentVectors = reshape(tangentVectors, size(tangentVectors,1)*size(tangentVectors,2),...
    size(tangentVectors,3)); 

testImages = reshape(testImages, size(testImages, 1) * size(testImages, 2), size(testImages, 3));

testImageTangentVectors=[testImages(:,:);tangentVectors(:,:)];
display 'preforming knn search'
[IDX, D] = knnsearch(trainImageTangentVectors(:,:)',testImageTangentVectors(:,:)', 'K', k_best,...
        'Distance', @tangentDistance2);
y = mode(trainLabels(IDX),2);
accuracy = nnz(y==testLabels)/num_test_images;
fprintf('Test set accuracy: %f\n',accuracy);