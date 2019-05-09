load TrainingLabels.mat
load FaceClassifier.mat
imgPath = '/Users/Jeff/Documents/MATLAB/TRAINING_NORM2/';

count = 0;
misses = zeros(1,80);
for person = 385:480
    currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", person, ".png"));
    resizedImage = imresize(currentImage, [90 90]);
%     fudgeFactor = 1;
%     try
%         [~, threshold] = edge(normalize8(rgb2gray(resizedImage)), 'sobel');
%         BW1 = edge(normalize8(rgb2gray(resizedImage)),'sobel', threshold * fudgeFactor);
%     catch
%         [~, threshold] = edge(normalize8(resizedImage), 'sobel');
%         BW1 = edge(normalize8(resizedImage),'sobel', threshold * fudgeFactor);
%     end
%     [featureVector,hogVisualization] = extractHOGFeatures(BW1);
    queryFeatures = extractHOGFeatures(resizedImage);
    try
        [personLabel, score, cost] = predict(faceClassifier,queryFeatures);
        if personLabel == TrainingSTR(person).id
            
            count = count + 1;
        else
            misses(TrainingSTR(person).id) = misses(TrainingSTR(person).id) + 1;
            fprintf("Guess = %d, Actual = %d, Score = %d, Cost = %d\n", personLabel,TrainingSTR(person).id, score(personLabel), cost(personLabel));
        end
    catch
        fprintf("error\n");
    end 
end
fprintf("%d Correct out of %d, Accuracy = %d ", count, 96, count/96);

function Y=normalize8(X,mode);

%% default return value
Y=[];

%% Parameter check
if nargin==1
    mode = 1;
end

%% Init. operations
X=double(X);
[a,b]=size(X);

%% Adjust the dynamic range to the 8-bit interval
max_v_x = max(max(X));
min_v_x = min(min(X));

if mode == 1
    Y=ceil(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b))))*255);
elseif mode == 0
    Y=(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b)))));
else
    disp('Error: Wrong value of parameter "mode". Please provide either 0 or 1.')
end
end