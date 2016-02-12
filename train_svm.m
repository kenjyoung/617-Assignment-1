clear
%load mnist data
display 'loading data...'
images = loadMNISTImages('data/train-images-idx3-ubyte');
labels = loadMNISTLabels('data/train-labels-idx1-ubyte');
num_images = size(images,3);
fold_size = ceil(num_images/10);

display 'computing HOGs'
cellSize = 8 ;
hog = vl_hog(im, cellSize) ;

w=

[w, bias] = vl_svmtrain(single(x), ...
                        y, ...
                        lambda, ...
                        'Solver', 'sdca', ...
                        'BiasMultiplier', 1, ...
                        'verbose') ;