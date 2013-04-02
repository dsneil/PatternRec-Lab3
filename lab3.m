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

clear xVals; clear yVals; clear testPts; clear cont; clear c;

% Create Confusion matrix for 3 data sets
confMat2 = Utils.CreateConfusion(classes2, f2t);
confMat8 = Utils.CreateConfusion(classes8, f8t);
confMat32 = Utils.CreateConfusion(classes32, f32t);

% classify multf8 image
figure;
cimage = Utils.ImageClassifier(multf8, classes8);
colormap(gray);

% plot original image
figure;
imagesc(multim);
colormap(gray);