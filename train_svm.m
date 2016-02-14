clear
%load mnist data
display 'loading data...'
images = single(loadMNISTImages('data/train-images-idx3-ubyte'));
labels = single(loadMNISTLabels('data/train-labels-idx1-ubyte'));
num_images = size(images,3);
fold_size = ceil(num_images/10);

best_params = [0,0];
best_accuracy = 0;
val_curve = zeros(5,10);
numfeatures = 300;
display 'preforming cross validation'
for logCellSize = 2:5
    clear hogs
    cellSize = 2^logCellSize;
    display 'computing HOGs...'
    for i=1:num_images
        hog = vl_hog(images(:,:,i), cellSize, 'variant', 'dalaltriggs');
        hogs(:,i) = hog(:);
    end
    display 'computing features...'
    code_len = size(hogs,1);

    feature_weights = mvnrnd(zeros(numfeatures,code_len),eye(code_len));

    feature_biases = unifrnd(0,2*pi,numfeatures,1);
    x = sqrt(2/numfeatures)*cos(feature_weights*hogs+repmat(feature_biases,1,num_images));
    for Cd2=1:10
        C=2*Cd2;
        for i=1:10
            validation_set = (i-1)*fold_size+1:min(i*fold_size+1,num_images);
            num_val_images = size(validation_set,2);
            training_set = [1:(i-1)*fold_size,i*fold_size+1+1:num_images];
            x_train = x(:,training_set);
            labels_train = labels(training_set);
            x_val = x(:,validation_set);
            labels_val = labels(validation_set);
            display 'training svms...'
            for class=0:9
                y_train=double(labels_train==class);
                y_train(~y_train)=-1;
                lambda = 1 / (C * numel(y_train)) ;
                [w(class+1,:), b(class+1,:)] = vl_svmtrain(single(x_train), ...
                                                y_train, ...
                                                lambda, ...
                                                'BiasMultiplier', 1) ;
            end

            scores = w * x_val + repmat(b,1,size(x_val,2));

            [~,class] = max(scores,[],1);
            class = class'-1;
            accuracy = nnz(class==labels_val)/num_val_images;
            val_curve(logCellSize,Cd2)=val_curve(logCellSize,Cd2)+accuracy/10;
        end
            fprintf('Current accuracy:%f\n', accuracy);
    end
end
[accuracy_best,i] = max(val_curve(:));
[x, y] = ind2sub(size(val_curve),i);
cellSize=2^x;
C=2*y;
display 'cross validation complete';

display 'recomputing with best params';
display 'computing HOGs...';
clear hogs
for i=1:num_images
    hog = vl_hog(images(:,:,i), cellSize, 'variant', 'dalaltriggs');
    hogs(:,i) = hog(:);
end
display 'computing features...'
code_len = size(hogs,1);
feature_weights = mvnrnd(zeros(numfeatures,code_len),eye(code_len));
feature_biases = unifrnd(0,2*pi,numfeatures,1);
x = sqrt(2/numfeatures)*cos(feature_weights*hogs+repmat(feature_biases,1,num_images));

for class=0:9
    y=double(labels==class);
    y(~y)=-1;
    lambda = 1 / (C * numel(y)) ;
    [w(class+1,:), b(class+1,:)] = vl_svmtrain(single(x), ...
                                    y, ...
                                    lambda, ...
                                    'BiasMultiplier', 1) ;
end

fprintf('best params are %d,%d with accuracy %f\n',cellSize, C, accuracy_best);
save 'train_svm_results' 'val_curve' 'w' 'b' 'feature_biases' 'feature_weights' 'numfeatures' 'cellSize' 'C'