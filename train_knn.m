clear
%load mnist data
display 'loading data...'
images = loadMNISTImages('data/train-images-idx3-ubyte');
images = images(:,:,1:1000);
labels = loadMNISTLabels('data/train-labels-idx1-ubyte');
num_images = size(images,3);
fold_size = ceil(num_images/10);

%compute tangent vectors
display 'computing tangent vectors'
tangentVectors = TangentVectors(images);
tangentVectors = reshape(tangentVectors, size(tangentVectors,1)*size(tangentVectors,2),...
    size(tangentVectors,3)); 

images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));

trainImageTangentVectors=[images(:,:);tangentVectors(:,:)];

display 'preform cross validation'
k_best = 0;
accuracy_best = 0;
val_curve = [];
for k=1:20
   accuracy = 0;
   for i=1:10
       validation_set = (i-1)*fold_size+1:min(i*fold_size+1,num_images);
       training_set = [1:(i-1)*fold_size,i*fold_size+1+1:num_images];
       val_labels =  labels(validation_set);
       [IDX, D] = knnsearch(trainImageTangentVectors(:,training_set)',...
        trainImageTangentVectors(:,validation_set)', 'K', k,...
        'Distance', @tangentDistance2);
       y = mode(labels(IDX),2);
       accuracy = accuracy + nnz(y==val_labels)/size(validation_set,2);
   end
   accuracy = accuracy/10;
   val_curve(k) = accuracy;
   fprintf('k=%d, accuracy=%f\n',k, accuracy);
   if(accuracy>accuracy_best)
       accuracy_best = accuracy;
       k_best = k;
   end
end
display 'cross validation complete';
fprintf('best k value is %d, with accuracy %f\n', k_best, accuracy_best);
save('train_knn_results', 'k_best', 'accuracy_best', 'trainImageTangentVectors', 'val_curve');

