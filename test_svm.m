clear
display 'loading trained model'
load('train_svm_results')
display 'loading test data'
images = single(loadMNISTImages('data/t10k-images-idx3-ubyte'));
labels = single(loadMNISTLabels('data/t10k-labels-idx1-ubyte'));
num_images = size(images,3);

for i=1:num_images
	hog = vl_hog(images(:,:,i), cellSize);
    hogs(:,i) = hog(:);
end

x = cos(feature_weights*hogs+repmat(feature_biases,1,num_images));

scores = w * x + repmat(b,1,num_images);

[~,class] = max(scores,[],1);
accuracy = nnz(class'==labels)/num_images