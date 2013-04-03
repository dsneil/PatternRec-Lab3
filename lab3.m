% Lab 3

clear all;
close all;

load('data/feat.mat');

figure;
classes2 = Utils.createClasses(f2); % create classes for the n=2 data;
classes8 = Utils.createClasses(f8); % create classes for the n=8 data;
classes32 = Utils.createClasses(f32); % create classes for the n=32 data;

[xVals, yVals, testPts, cont] = Utils.createGrid(0.001, 0, classes8);

for c=1:length(classes8)
	Utils.plotClass(classes8(c));
end

% create the MICD classifier for n=8
hold on;
MICDCont = Utils.MICDClassifier('k', xVals, yVals, testPts, cont, classes8);

clear xVals; clear yVals; clear testPts; clear c;

% Create Confusion matrix for 3 data sets
confMat2 = Utils.CreateConfusion(classes2, f2t);
confMat8 = Utils.CreateConfusion(classes8, f8t);
confMat32 = Utils.CreateConfusion(classes32, f32t);

error2 = trace(confMat2)/sum(sum(confMat2));
error8 = trace(confMat8)/sum(sum(confMat8));
error32 = trace(confMat32)/sum(sum(confMat32));

% classify multf8 image
figure;
cimage = Utils.ImageClassifier(multf8, classes8);
% colormap(gray);

% plot original image
figure;
imagesc(multim);
% colormap(gray);