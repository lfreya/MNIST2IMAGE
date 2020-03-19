clear;
tic;
img_train = loadMNISTImages('train-images.idx3-ubyte');
label_train = loadMNISTLabels('train-labels.idx1-ubyte');
img_test = loadMNISTImages('t10k-images.idx3-ubyte');
label_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
% mkdir('train');
% mkdir('test');
% for i=0:9
% mkdir(strcat('train',num2str(i)));
% end
% for i=0:9
% mkdir(strcat('test',num2str(i)));
% end

for i=1:60000
img=reshape(img_train(:,i),28,28);
%可以转换为任意图像格式.jpg .png .bmp
imgName=strcat('train',num2str(label_train(i)),'',num2str(label_train(i)),'',num2str(i));
fileName=num2str(label_train(i));
imwrite(img,['./jpg/',fileName,'/',imgName,'.jpg']);
end
for i=1:10000
img=reshape(img_test(:,i),28,28);
imgName=strcat('test',num2str(label_test(i)),'',num2str(label_test(i)),'',num2str(i));
fileName=num2str(label_test(i));
imwrite(img,['./jpg/',fileName,'/',imgName,'.jpg']);
end
toc;

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);
fclose(fp);
% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in', filename, '']);
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp);
end
