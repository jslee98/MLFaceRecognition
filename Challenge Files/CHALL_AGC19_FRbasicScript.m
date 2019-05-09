% Basic script for Face Recognition Challenge
% --------------------------------------------------------------------
% AGC Challenge 2019 
% Universitat Pompeu Fabra
% By Jeffrey Lee & Oriol Resina
%

% Load challenge Training data
load AGC19_Challenge3_Training.mat % Insert testing file here
load FaceClassifier.mat % Ensure FaceClassifier.mat is in working directory

% Provide the path to the input images, for example 
% 'C:\AGC_Challenge_2019\images\'
imgPath = '';

% Initialize results structure
AutoRecognSTR = struct();

% Initialize timer accumulator
total_time = 0;

% count = 1;
my_FRmodel = faceClassifier;

% Process all images in the Testing set
for j = 1 : 1200 % Correct Length here
    A = imread( sprintf('%s%s', imgPath, ...
        AGC19_Challenge3_TRAINING(j).imageName ));   % ensure this is the correct Struct
    fprintf("Processing %d...\n", j);
    
        tic;
        
%         used for 5-fold testing
%         if j <= 240
%             my_FRmodel = faceClassifier1;
%         elseif j <= 480
%             my_FRmodel = faceClassifier2;
%         elseif j <= 720
%             my_FRmodel = faceClassifier3;
%         elseif j <= 960
%             my_FRmodel = faceClassifier4;
%         else
%             my_FRmodel = faceClassifier5;
%         end
        
        autom_id = my_face_recognition_function( A, my_FRmodel );
        
        % Update total time
        tt = toc;
        total_time = total_time + tt;

    % Store the detection(s) in the results structure
    AutoRecognSTR(j).id = autom_id;
end
   
% Compute detection score
FR_score = CHALL_AGC19_ComputeRecognScores(...
    AutoRecognSTR, AGC19_Challenge3_TRAINING);

% Display summary of results
fprintf(1, '\nF1-score: %.2f%% \t Total time: %dm %ds\n', ...
    100 * FR_score, int16( total_time/60),...
    int16(mod( total_time, 60)) );

%% Returns locations of faces in image A
function output = MyFaceDetectionFunction( A )
faceDetector = vision.CascadeObjectDetector;
faceDetector.MergeThreshold = 8;
bbox = step(faceDetector, A);
output = bbox;
end

%% Finds biggest two faces
function output = find_biggest_two_faces(facebox)
biggestfaces = zeros(2,4);
maxwidth = 0;
maxwidth2 = 0;
for i = 1 : size(facebox,1)
    boxwidth = facebox(i,3);
    if boxwidth > maxwidth
        biggestfaces(2,:) = biggestfaces(1,:);
        biggestfaces (1,:) = facebox(i,:);
        maxwidth2 = maxwidth;
        maxwidth = boxwidth;
    elseif boxwidth > maxwidth2
        biggestfaces (2,:) = facebox(i,:);
        maxwidth2 = boxwidth;
    end
end
output = biggestfaces;
end

%% Returns ID of input image A, or -1 if not found
function output = my_face_recognition_function(A, myFRmodel)

% Detect faces
faceboxes = MyFaceDetectionFunction(A);

% If the result is empty, no faces found
if isempty(faceboxes)
    output = -1;
else
    if size(faceboxes,1) > 1
        faceboxes = find_biggest_two_faces(faceboxes);
    end
    facebox = faceboxes(1,:);
    
    % Process face image
    try
        grayscaleImage = rgb2gray(A);
    catch
        grayscaleImage = A;
    end
    croppedFace = imcrop(grayscaleImage, facebox);
    resizedFace = imresize(croppedFace, [90 90]);
    queryFeatures = extractHOGFeatures(resizedFace);
    [faceid, score, cost] = predict(myFRmodel, queryFeatures);
    thresh = -0.203;
    
    % If the faceid score is below the threshold and there is a second face
    % Test second face
    if score(faceid) <= thresh && size(faceboxes, 1) > 1
        facebox = faceboxes(2,:);
        croppedFace = imcrop(grayscaleImage, facebox);
        resizedFace = imresize(croppedFace, [90 90]);
        queryFeatures = extractHOGFeatures(resizedFace);
        [faceid, score, cost] = predict(myFRmodel, queryFeatures);   
    end
    
    % Ensure faceid score is above threshold
    if score(faceid) > thresh
        output = faceid;
    else
        output = -1;
    end
   
end
end



