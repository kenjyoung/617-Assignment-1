clear
%load mnist data
display 'loading data...'
images = loadMNISTImages('data/train-images-idx3-ubyte');
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
k_max=50;
val_curve = zeros(k_max,1);
for i=1:10
   validation_set = (i-1)*fold_size+1:min(i*fold_size+1,num_images);
   training_set = [1:(i-1)*fold_size,i*fold_size+1+1:num_images];
   val_labels =  labels(validation_set);
   [IDX, D] = knnsearch(trainImageTangentVectors(:,training_set)',...
    trainImageTangentVectors(:,validation_set)', 'K', k_max,...
    'Distance', @tangentDistance2);
   while(size(IDX,2)>0)
       k=size(IDX,2);
       y = mode(labels(IDX),2);
       accuracy = nnz(y==val_labels);
       val_curve(k) = val_curve(k)+accuracy/(size(validation_set,2)*10);
       [~,worst] = max(D,[],2);
       D(:,worst)=[];
       IDX(:,worst)=[];
   end
   fprintf('%d percent complete\n',i*10);
end
val_curve
[accuracy_best,k_best] = max(val_curve);

display 'cross validation complete';
fprintf('best k value is %d, with accuracy %f\n', k_best, accuracy_best);
save('train_knn_results', 'k_best', 'accuracy_best', 'trainImageTangentVectors', 'val_curve');

