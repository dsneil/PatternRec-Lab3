classdef Utils
    methods (Static)

        function plotClass(Class)
            % Simplifies plotting process. Plots on existing figure.

            hold on;
            scatter(Class.Cluster(:,1), Class.Cluster(:,2), 5,'filled');
        end

		function [xVals, yVals, testPoints, zeroGrid] = createGrid(stepSize, buffer, classes)
            % -Creates a grid of zeros to be used in classification procedures
            % -Returns a list of xVals and yVals containing all points of eval
            % -All combinations are contained in testPoints
            % -ZeroGrid is the contour to be populated in classification
            % --
            % stepSize = resolution of zeroGrid
            % buffer = padding around feature space
            % varargin = list of classes used to create grid

            mixedVals = [];
            for k = 1 : length(classes)
                %appends all cluster data into mixedVals matrix
                mixedVals = [mixedVals; classes(k).Cluster];
            end
            
            %extracts min/max x and y from mixed data
            mins = min(mixedVals);
            maxs = max(mixedVals);
            
            xVals = mins(:,1)-buffer:stepSize:maxs(:,1)+buffer;
            yVals = mins(:,2)-buffer:stepSize:maxs(:,2)+buffer;
            
            [x,y] = meshgrid(xVals, yVals);
            c = cat(2,x',y');
            testPoints = reshape(c,[],2);
            
            zeroGrid = zeros(length(xVals),length(yVals));
        end

        function data = createClasses(rawData)
            % Takes a matrix of features with class labels and creates mtx of classes
            % --
            % rawData = (4xn) matrix of data
            %   - 1xn: feature in first dimension
            %   - 2xn: feature in second dimension
            %   - 3xn: class label
            %   - 4xn: sample number

            numCs = max(rawData(3,:));
            data = [];
            for k=1:numCs
                cData = rawData(1:2,rawData(3,:)==k);
                % cData = reshape(cData, length(cData(1,:)), length(cData(:,1)));

                cls = classData(cData', 'k');
                cls.Mean = Utils.learnMean(cls);
                cls.Cov = Utils.learnCovariance(cls);
                cls.InvCov = inv(cls.Cov);

                % Utils.plotClass(cls);

                data = [data cls];
            end
        end

        function mu = learnMean(class)
            % Learns the mean of a data set
            mu = ((1/length(class.Cluster))*sum(class.Cluster));
        end

        function var = learnVariance(class)
            % Learns the variance of a data set
            if (isempty(class.Var) == 1)
                temp = 0;
                data = class.Cluster;
                for k=1:length(data),
                    temp = temp + (data(k,:)-class.Mean)*(data(k,:)-class.Mean)';
                end

                var = ((1/(length(class.Cluster)*length(data)))*temp);
            else
                var = dvar;
            end
        end

        function cov = learnCovariance(class)
            % Learns the covariance matrix of a data set
            temp = [0 0; 0 0]; %set defaults
            data = class.Cluster;

            for k=1:length(data),
                temp = temp + (data(k,:)-class.Mean)'*(data(k,:)-class.Mean);
            end

            cov = ((1/(length(data)))*temp);
        end

        function eD = eucD(p1, p2)
            eD = sqrt(sum((p1 - p2) .^ 2));
        end

        %--------------------
        % CLASSIFIERS
        %--------------------

        function confmat = CreateConfusion(classes, testD)
            testClasses = Utils.createClasses(testD);
            confmat = [];
            for k = 1:length(testClasses)
                confmat = [confmat; Utils.MICDConfusion(testClasses(k).Cluster, classes)];
            end
        end

        function conf = MICDConfusion(testPts, classes)
            numVar = length(classes); numPts = length(testPts(:,1));
            dists = []; conf = zeros(1,numVar);

            for k = 1:numPts
                for s = 1:numVar
                    dif = testPts(k,:) - classes(s).Mean;
                    transVals = dif*classes(s).InvCov*dif';

                    dists = [dists transVals];
                end

                [~, minClass] = min(dists);
                conf(1,minClass) = conf(1,minClass) +1; %conf mat histogram
                dists = [];
            end
        end

        function cont = MICDDecision(numXs, testPts, cont, classes)
            numVar = length(classes);
            dists = [];
            xIndex = 1; yIndex = 1;

            for k = 1: length(testPts(:,1))
                for s = 1 : numVar
                    dif = testPts(k,:) - classes(s).Mean; %(x-m)
                    transVals = dif*classes(s).InvCov*dif'; % (x-m)*invS*(x-m)'
                
                    dists = [ dists transVals];
                end
                [~, minClass] = min(dists);
                cont(xIndex,yIndex) = minClass;
                dists = [];
                
               if(xIndex == numXs)
                    xIndex = 1;
                    yIndex = yIndex +1;
                else
                    xIndex = xIndex +1;
                end
            end
        end

        function cimage = ImageClassifier(imageData, classes)

            feature1 = imageData(:,:,1); feature2 = imageData(:,:,2);
            compFeatures = [feature1(:), feature2(:)];
            cont = zeros(length(imageData(:,1,1)), length(imageData(1,:,1)));
            numXs = length(imageData(1,:,1));

            cimage = Utils.MICDDecision(numXs, compFeatures, cont, classes);

            imagesc(cimage);
        end

        function finCont = MICDClassifier(color, xVals, yVals, testPts, cont, classes)

            cont = Utils.MICDDecision(length(xVals), testPts, cont, classes);
            finCont = cont';
            [c, h] = contour(xVals,yVals, finCont, color);
            %ch = get(h,'child'); alpha(ch,0.05)
            
        end
    end
end