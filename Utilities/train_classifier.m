load TrainingLabels.mat
imgPath = '/Users/Jeff/Documents/MATLAB/TRAINING_FINAL/';

%% Train large classifier for testing
trainingFeatures = zeros(540,3600);
trainingLabels = zeros(1,540);

for featureCount = 1 : 540
    currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
    [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
    trainingFeatures(featureCount,:) = extractHOGFeatures(currentImage);
    trainingLabels(featureCount) = TrainingSTR(featureCount).id;
    % visualize hog
%     figure;
%     subplot(1,2,1);
%     imshow(currentImage);
%     subplot(1,2,2);
%     plot(hogVisualization);
end

faceClassifier = fitcecoc(trainingFeatures, trainingLabels);
save('FaceClassifier', 'faceClassifier');


%% Below is the code to train 5-fold models
%
% trainingFeatures = zeros(381,3600);
% trainingLabels = zeros(1,381);
% count = 1;
% 
% fprintf("Training 1...\n");
% 
% for featureCount = 100 : 191
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 192 : 275
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 276 : 389
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 390 : 480
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(featureCount,:) = extractHOGFeatures(currentImage);
%     trainingLabels(featureCount) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% 
% faceClassifier1 = fitcecoc(trainingFeatures, trainingLabels);
% save('Classifier1', 'faceClassifier1');
% 
% trainingFeatures = zeros(388,3600);
% trainingLabels = zeros(1,388);
% count = 1;
% fprintf("Training 2...\n");
% for featureCount = 1 : 99
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 192 : 275
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 276 : 389
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 390 : 480
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(featureCount,:) = extractHOGFeatures(currentImage);
%     trainingLabels(featureCount) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% 
% 
% faceClassifier2 = fitcecoc(trainingFeatures, trainingLabels);
% save('Classifier2', 'faceClassifier2');
% 
% trainingFeatures = zeros(396,3600);
% trainingLabels = zeros(1,396);
% count = 1;
% fprintf("Training 3...\n");
% for featureCount = 1 : 99
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 100 : 191
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 276 : 389
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 390 : 480
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(featureCount,:) = extractHOGFeatures(currentImage);
%     trainingLabels(featureCount) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% 
% 
% faceClassifier3 = fitcecoc(trainingFeatures, trainingLabels);
% save('Classifier3', 'faceClassifier3');
% 
% trainingFeatures = zeros(366,3600);
% trainingLabels = zeros(1,366);
% count = 1;
% fprintf("Training 4...\n");
% for featureCount = 1 : 99
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 100 : 191
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 192 : 275
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% 
% for featureCount = 390 : 480
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(featureCount,:) = extractHOGFeatures(currentImage);
%     trainingLabels(featureCount) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% 
% faceClassifier4 = fitcecoc(trainingFeatures, trainingLabels);
% save('Classifier4', 'faceClassifier4');
% 
% trainingFeatures = zeros(389,3600);
% trainingLabels = zeros(1,389);
% count = 1;
% fprintf("Training 5...\n");
% for featureCount = 1 : 99
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 100 : 191
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 192 : 275
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% for featureCount = 276 : 389
%     currentImage = imread(sprintf("%s%s%d%s", imgPath, "img", featureCount, ".png"));
%     [featureVector,hogVisualization] = extractHOGFeatures(currentImage);
%     trainingFeatures(count,:) = extractHOGFeatures(currentImage);
%     trainingLabels(count) = TrainingSTR(featureCount).id;
%     count = count + 1;
% end
% 
% faceClassifier5 = fitcecoc(trainingFeatures, trainingLabels);
% save('Classifier5', 'faceClassifier5');