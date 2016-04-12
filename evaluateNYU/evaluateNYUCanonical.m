%evaluates a technique

imageCount = 1449;

forceRecompute = 1;
downsampleFactor = 1;

pixelErrorSubsample = 1;
%pixelErrorSubsample = 512;

%resultsLoader = @(x)(loadSVMBaseline(x));
%loadPrior(x));

subsample = @(x)(x(1:10:end));

evalFunctions = {
                {'Mean', @(x)(rad2deg(mean(x)))},
                {'Median', @(x)(rad2deg(median(subsample(x))))},
                {'RMSE',@(x)(rad2deg(sqrt(mean(x.^2))))}
                };

%evalFunctions = {};
degs = [90 / 8, 90 / 4, 90 / 3, 90 / 2];
%degs = [90 / 4];
threshes = deg2rad(degs);

for i=1:numel(threshes)
    evalFunctions{end+1} = {sprintf('Bad Pixels %2.2f',degs(i)), @(x)((100 - 100*mean(x > threshes(i))))};
end


%for getting at the various normal map formats
%from a loaded mat file 
gtLoad = @(m)(cat(3,m.nx,m.ny,m.nz));
threshLoad = @(m)(m.normalMaps{end});
singleLoad = @(m)(m.normalMap);
singleLoadNew = @(m)(m.Nm);
packedLoadRect = @(m)(m.nd.rectifiedDenseMaps{1});
packedLoadUnrect = @(m)(m.nd.denseMaps{1});

gload = @(m)(m.N);
pload = @(m)(m.nmap);
sload1 = @(m)(m.nmapsFill{1});
sload = @(m)(m.nmapsFill{3});
sload10 = @(m)(m.nmapsFill{10});

dload = @(m)(decrop(resizeNormals(m.N3,[427,561])));

validMaskLoad = @(m)(m.wasPredicted{end});

resultsLoaders = {
    {   'soft-assignment',
         @(i)(sprintf('/home/yu/seg_proj/FCN/Sec_2/test_res/rgb_%06d.mat',i)),
        dload,
        []
    },

};


techniqueCount = numel(resultsLoaders);
numEvals = size(evalFunctions,1);

results = zeros(imageCount,techniqueCount,numEvals);

accum = cell(imageCount,techniqueCount);
accumAngles = cell(imageCount,techniqueCount);

parfor i=1:imageCount

    if ~exist(sprintf('/home/yu/seg_proj/FCN/Sec_2/test_res/rgb_%06d.mat',i))
        continue;
    end

    fprintf('%d\n',i);
    gtData = loadGT(i);

    validMask = gtData.depthValid(1:downsampleFactor:end,1:downsampleFactor:end);

    for j=1:techniqueCount
        filename = feval(resultsLoaders{j}{2},i);
        cache = [filename '.errcache'];
            
        errorMap = 0; computeErrorMap = 1;

        if exist(cache)
            filenameData = dir(filename);
            cacheData = dir(cache);
            if cacheData.datenum > filenameData.datenum
                computeErrorMap = 0;
                errorMapData = load(cache, '-mat');
                errorMap = errorMapData.errorMap; 
            end 
        end 

        if (forceRecompute)
            computeErrorMap = 1;
        end

        if(computeErrorMap)
            predicted = feval(resultsLoaders{j}{3},load(filename));
            predictedDiv = sum(predicted.^2,3).^0.5;
            predicted = predicted ./ repmat(predictedDiv + eps, [1, 1, 3]); 
            errorMap = computeNormalAngleDiffMap(predicted, gtData);
            %downsample
            errorMap = errorMap(1:downsampleFactor:end,1:downsampleFactor:end);
            %save the error map to the cache
%            saveToMat(cache, 'errorMap', errorMap);
        end

        evalMask = ones(size(errorMap));

        if(~isempty(resultsLoaders{j}{4}))
            %if we have to load a mask, load it
            predictedMask = feval(resultsLoaders{j}{4},load(filename));
            evalMask = predictedMask(1:downsampleFactor:end,1:downsampleFactor:end); 
        end


        toEvaluate = find(validMask .* evalMask);

        knownAngleDiffs = errorMap(find(validMask .* evalMask));
        knownAngleDiffs = knownAngleDiffs(randperm(numel(knownAngleDiffs)));

        accum{i,j} = knownAngleDiffs(1:pixelErrorSubsample:end);
        
        for k=1:numEvals
            results(i,j,k) = feval(evalFunctions{k}{2},knownAngleDiffs);
        end
        
    end
end



%results averaging over images and pixels
endResultsImage = cell(techniqueCount,numEvals); 
endResultsPixel = cell(techniqueCount,numEvals);

techniqueNames = cell(techniqueCount,1);
evalNames = cell(1,numEvals);

for k=1:numEvals
    evalNames{k} = evalFunctions{k}{1};
end
for j=1:techniqueCount
    techniqueNames{j} = resultsLoaders{j}{1};
end

fprintf('All Done\n');

alphas = [0.05];
for ai=1:numel(alphas)
opts = statset('UseParallel','always');
for k=1:numEvals
    fprintf('======\n%s\n=====\n',evalFunctions{k}{1});
    for j=1:techniqueCount
        evalF = @(inds)(feval(evalFunctions{k}{2},cat(1,accum{inds,j})));
        pixelValues = cat(1, accum{:,j});
        fprintf('\n%s\n',resultsLoaders{j}{1});
        if(0)
            meanPixelCI = bootci(10000,{evalF,1:size(accum,1)},'Options',opts,'alpha',alphas(ai));
        else
            meanImageCI = [0,0]; meanPixelCI = [0,0];
        end
        overPixels = feval(evalFunctions{k}{2},pixelValues);
        fprintf('  Over pixels:      %2.3f:  95%% CI: [%2.3f, %2.3f]\n', overPixels, meanPixelCI(1), meanPixelCI(2));
        fprintf('  \\bci{%2.1f}{%2.1f}{%2.1f}\n',overPixels,meanPixelCI(1), meanPixelCI(2));
        endResultsPixel{j,k} = [meanPixelCI(1), overPixels, meanPixelCI(2)];
    end
    fprintf('\n');
end
end


