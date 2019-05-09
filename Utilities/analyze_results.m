load Results2.mat

numTP = 0;
numNotInDB = 0;
numFalseDetections = 0;

tpMax = -10;
tpMin = 10;
nidMax = -10;
nidMin = 10;
fdMax = -10;
fdMin = 10;

tpSum = 0;
nidSum = 0;
fdSum = 0;

thresh = -.00730;

for i = 1 : length(ResSTR)
    if ResSTR(i).detected ~= -1 && ResSTR(i).detected == ResSTR(i).real
        if ResSTR(i).score < thresh
            fprintf("Dropped tp\n");
        end
        numTP = numTP + 1;
        tpSum = tpSum + ResSTR(i).score;
        if ResSTR(i).score > tpMax
            tpMax = ResSTR(i).score;
        elseif ResSTR(i).score < tpMin
            tpMin = ResSTR(i).score;
        end
    elseif ResSTR(i).detected > -1 && ResSTR(i).real == -1
        if ResSTR(i).score < thresh
            fprintf("Dropped nid\n");
        end
        numNotInDB = numNotInDB + 1;
        nidSum = nidSum + ResSTR(i).score;
        if ResSTR(i).score > nidMax
            nidMax = ResSTR(i).score;
        elseif ResSTR(i).score < nidMin
            nidMin = ResSTR(i).score;
        end
    elseif ResSTR(i).detected > -1 && ResSTR(i).real > -1
        if ResSTR(i).score < thresh
            fprintf("Dropped tp\n");
        end
        fdSum = fdSum + ResSTR(i).score;
        numFalseDetections = numFalseDetections + 1;
        if ResSTR(i).score > fdMax
            fdMax = ResSTR(i).score;
        elseif ResSTR(i).score < fdMin
            fdMin = ResSTR(i).score;
        end
    end
    
end


fprintf("Num True Positives = %d, Avg Score = %d, Max = %d, Min = %d\n", numTP, tpSum/numTP, tpMax, tpMin);
fprintf("Num Not In DB = %d, Avg Score = %d, Max = %d, Min = %d\n", numNotInDB, nidSum/numNotInDB, nidMax, nidMin);
fprintf("Num False Positives = %d, Avg Score = %d, Max = %d, Min = %d\n", numFalseDetections, fdSum/numFalseDetections, fdMax, fdMin);