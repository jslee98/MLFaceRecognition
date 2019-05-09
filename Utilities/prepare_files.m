load AGCTraining2.mat
imgPath = '/Users/Jeff/Documents/MATLAB/TRAINING/';
newImgPath = '/Users/Jeff/Documents/MATLAB/TRAINING_FINAL/';
TrainingSTR = struct();
faceCount = 1;

for j = 1 : length( AGC19_Challenge3_TRAINING )
    faceID = AGC19_Challenge3_TRAINING(j).id;
    if (faceID > -1)
        fprintf("prepping %d...\n", j);
        faceImage = imread(sprintf('%s%s', imgPath, AGC19_Challenge3_TRAINING(j).imageName));
        facebox = MyFaceDetectionFunction(faceImage);
        try
            grayscaleImage = rgb2gray(faceImage);
        catch
            % Face is already grayscale
            grayscaleImage = faceImage;
        end

        if size(facebox,1) > 1
            biggestbox = find_biggest_face(facebox);
            croppedFace = imcrop(grayscaleImage, biggestbox);
        elseif size(facebox,1) == 0 || facebox(1) == 0 && facebox(3) == 0
            % If no face is detected with our algorithm, 
            % we use the given facebox
            facebox = AGC19_Challenge3_TRAINING(j).faceBox;
            facebox = find_biggest_face(facebox);
            facebox(1,3) = facebox(1,3) - facebox(1,1);
            facebox(1,4) = facebox(1,3);
            croppedFace = imcrop(grayscaleImage, facebox);
        else
            croppedFace = imcrop(grayscaleImage, facebox); 
        end
        resizedImage = imresize(croppedFace, [90 90]);
        %imshow(croppedFace);
        newFile = sprintf('%s%s%d%s', newImgPath, "img", faceCount, ".png");
        imwrite(resizedImage, newFile);
        TrainingSTR(faceCount).id = faceID;
        TrainingSTR(faceCount).originalFile = AGC19_Challenge3_TRAINING(j).imageName;
        TrainingSTR(faceCount).newName = newFile;
        
        faceCount = faceCount + 1;
    end
end

function output = MyFaceDetectionFunction( A )

% Run face detector on image A
faceDetector = vision.CascadeObjectDetector;
faceDetector.MergeThreshold = 8;
bbox = step(faceDetector, A);

% We know in training data we only want the biggest face
if size(bbox,1) > 1
    output = find_biggest_face(bbox);
else
    output = bbox;
end
end

function output = find_biggest_face(facebox)
output = zeros(1,4);
maxwidth = 0;
for i = 1 : size(facebox,1)
    boxwidth = facebox(i,3);
    if boxwidth > maxwidth
        output = facebox(i,:);
        maxwidth = boxwidth;
    end
end
end