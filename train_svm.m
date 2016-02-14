clear
%load mnist data
display 'loading data...'
images = loadMNISTImages('data/train-images-idx3-ubyte');
images = single(images(:,:,1:200));
labels = loadMNISTLabels('data/train-labels-idx1-ubyte');
labels = single(labels(1:200));
num_images = size(images,3);
fold_size = ceil(num_images/10);

display 'computing HOGs'
cellSize = 8;
numfeatures = 100;
C = 10;


for i=1:num_images
	hog = vl_hog(images(:,:,i), cellSize);
    hogs(:,i) = hog(:);
end

code_len = size(hogs,1);

feature_weights = mvnrnd(zeros(numfeatures,code_len),eye(code_len));
feature_weights = feature_weights*(2*pi)^(-code_len/2);

feature_biases = unifrnd(0,2*pi,1,numfeatures)';

x = cos(feature_weights*hogs+repmat(feature_biases,1,num_images));

for class=0:9
	y=double(labels==class)';
    y(~y)=-1;
    
    lambda = 1 / (C * numel(y)) ;
	[w(class+1,:), b(class+1,:)] = vl_svmtrain(single(x), ...
	                        y, ...
	                        lambda, ...
	                        'Solver', 'sdca', ...
	                        'BiasMultiplier', 1) ;
end
save 'train_svm_results' 'w' 'b' 'feature_biases' 'feature_weights' 'cellSize' 'C'