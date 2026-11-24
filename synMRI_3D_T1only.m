% tried to input tumorseg - if input all three labels useful, if you input
% just the tumor ROI not useful - decided to use no input tumorSeg.
%T1 inputl only
% addition of uncertainty aware training blurs the image no use
% adding a spatial frequency loss
% ---------------------------------------------------------------
% Clean GPU / parallel pool
% ---------------------------------------------------------------
pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);
end
gpuDevice([]);
%% path
addpath 'Y:\Brachy\Final_Program'
addpath 'Y:\Matlab_Addon_Tools\imshow3Dfull.m'
addpath  Y:\Matlab_Addon_Tools\imoverlay\
%% training data
for nn=1:3
    if nn==1
        targetModality = 't1c';       
    elseif nn==2
        targetModality = 't2';        
    elseif nn==3
        targetModality = 't2f';        
    end
    dstI = 'Y:\Syn_MRI\BRaTS 2021\MatData\Train\3D\Input';
    dstO = 'Y:\Syn_MRI\BRaTS 2021\MatData\Train\3D\Response';

    imdsInp = imageDatastore(dstI,'FileExtensions','.mat',...
        'ReadFcn',@(x) matReadImg(x));
    imdsSeg = imageDatastore(dstI,'FileExtensions','.mat',...
        'ReadFcn',@(x) matReadSeg(x));
    imdsRef = imageDatastore(dstO,'FileExtensions','.mat',...
        'ReadFcn',@(x) matReadRef(x,nn));

    % ---------------------------------------------------------------
    % Validation data
    % ---------------------------------------------------------------
    dstI = 'Y:\Syn_MRI\BRaTS 2021\MatData\Valid\3D\Input';
    dstO = 'Y:\Syn_MRI\BRaTS 2021\MatData\Valid\3D\Response';

    imdsValInp = imageDatastore(dstI,'FileExtensions','.mat',...
        'ReadFcn',@(x) matReadValImg(x));
    imdsValSeg = imageDatastore(dstI,'FileExtensions','.mat',...
        'ReadFcn',@(x) matReadSeg(x));
    imdsValRef = imageDatastore(dstO,'FileExtensions','.mat',...
        'ReadFcn',@(x) matReadRef(x,nn));

    dsVal = combine(imdsValInp,imdsValRef);

    % Create minibatchqueue from the combined datastore
    mbqVal = minibatchqueue(dsVal,...
        MiniBatchSize=1,...
        MiniBatchFcn=@(x,y) deal(cat(5,x{:}), cat(5,y{:})),...
        PartialMiniBatch="discard",...
        MiniBatchFormat=["SSSCB","SSSCB"],...
        OutputEnvironment="gpu");
 
    % Model
    miniBatchSize = 6;
    patchSize = [80 80 80]; % make sure your image size is bigger than patchSize
    numTrainVolumes = numel(imdsInp.Files);
    executionEnvironment = "gpu";

    intNumFilt = 32;
    % create generator % multihead deep supervision not used
    genX2Y = UNET_SkipAtten_SWINBN(patchSize,intNumFilt,3,5,1,1); % (imageSize,initialNumFilters,numDownSamp,numResBlock,add_atten,numOPChannel)
    %Create 3D discriminators
    disc1 = PatchGAN3DDiscriminator(patchSize);
    disc2 = PatchGAN3DDiscriminator(patchSize/2);
    model_name = targetModality + "-unetSwinT1only-epoch";
    % Training
    numEpochs =150;
    trailAvgGen = []; trailAvgSqGen = [];
    trailAvgDisc1 = []; trailAvgSqDisc1 = [];
    trailAvgDisc2 = []; trailAvgSqDisc2 = [];
    gradDecayFac = 0.5; sqGradDecayFac = 0.999;
    %Training Model
    validFreq = 1000; epoch = 0; start = tic; iteration = 0;

    % learning Rate parameters
    initialLearnRate = 0.0001; 
    LearnRate = initialLearnRate;
    decayFactor = 0.5;       % Reduce LR by N% every decay step
    decayEveryEpochs = 50;   % Decay every N epochs
    updateLR = decayEveryEpochs;

    % monitor trainging progress
    monitor = trainingProgressMonitor( ...
        Metrics="TrainingLoss");
    monitor.Info=["LearningRate","Epoch","Iteration","ExecutionEnvironment"];
    monitor.XLabel = "Iteration";
    monitor.Progress = 0;

    numObservationsTrain = numel(imdsInp.Files);
    numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
    numIterations = numEpochs*numIterationsPerEpoch;

    dataDir = fullfile('Y:\Syn_MRI\');
    % Create a directory to store checkpoints
    checkpointDir = fullfile(dataDir,"Models");
    if ~exist(checkpointDir,"dir")
        mkdir(checkpointDir);
    end

    % Initialize plots for training progress
    [figureHandle, tileHandle, imageAxes1, imageAxes2, imageAxes3, scoreAxesY, LossAxesY, ...
        lineScoreGenXToY, lineScoreDiscY, lineValLossX2Y] = initializeTrainingPlot_SupGAN();
    % Make sure figure is visible (especially in Live Script)
    set(figureHandle, 'Visible', 'on');

    % Chunk Buffer Configuration
    bufferSize = min(50, numel(imdsInp.Files));
    refreshEveryIters = 50;
    numVolumesTotal = numel(imdsInp.Files);

    % Initialize buffers
    [ramInp, ramRef, ramSeg] = refreshVolumeBuffer(imdsInp, imdsRef, imdsSeg, bufferSize, numVolumesTotal, nn);

    smoothedLoss = [];
    ValLoss = NaN;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;

        % Loop over mini-batches
        for i = 1:numIterationsPerEpoch
            if monitor.Stop, break; end
            iteration = iteration + 1;
            % Periodically reload buffer
            if mod(iteration, refreshEveryIters) == 0
                [ramInp, ramRef, ramSeg] = refreshVolumeBuffer(imdsInp, imdsRef, imdsSeg, bufferSize, numVolumesTotal,nn);
            end

            % Extract patches
            [Inp, OutPut, Seg] = extractTumorGuidedPatches(ramInp, ramRef, ramSeg, patchSize, miniBatchSize);
            Inp = dlarray(single(Inp), 'SSSCB');
            OutPut = dlarray(single(OutPut), 'SSSCB');
            Seg = dlarray(single(Seg), 'SSSCB');
            if executionEnvironment == "gpu"
                Inp = gpuArray(Inp);
                OutPut = gpuArray(OutPut);
                Seg = gpuArray(Seg);
            end

            % Calculate the loss and gradients
            [gradParamG, gradParamDisc1, gradParamDisc2, Loss, scores] = ...
                dlfeval(@modelGradients, Inp, OutPut, Seg, genX2Y, disc1, disc2,...
                targetModality);

            % Update parameters of generator
            [genX2Y, trailAvgGen, trailAvgSqGen] = adamupdate( ...
                genX2Y, gradParamG, trailAvgGen, trailAvgSqGen, iteration, ...
                LearnRate, gradDecayFac, sqGradDecayFac);

            % Update parameters of discriminator
            [disc1, trailAvgDisc1, trailAvgSqDisc1] = adamupdate( ...
                disc1, gradParamDisc1, trailAvgDisc1, trailAvgSqDisc1, iteration, ...
                LearnRate, gradDecayFac, sqGradDecayFac);

            [disc2, trailAvgDisc2, trailAvgSqDisc2] = adamupdate( ...
                disc2, gradParamDisc2, trailAvgDisc2, trailAvgSqDisc2, iteration, ...
                LearnRate, gradDecayFac, sqGradDecayFac);

            % Validation display
            if mod(iteration,validFreq) == 0 || iteration == 1
                % Reset Validation data
                if hasdata(mbqVal) == 0
                    reset(mbqVal);
                    shuffle(mbqVal);
                end
                [~, ValLoss] = displayGeneratedValImages(mbqVal, ...
                    imageAxes1, imageAxes2, imageAxes3,...
                    LossAxesY, lineValLossX2Y, genX2Y, iteration, patchSize);
                drawnow limitrate;
                ValLoss = round(ValLoss,3);
            end

            % Update the training progress - Record metrics.
            if iteration == 1
                smoothedLoss = Loss;
            else
                smoothedLoss = 0.95 * smoothedLoss + 0.05 * Loss;
            end
            recordMetrics(monitor, iteration, TrainingLoss=smoothedLoss);
            monitor.Info=["LearningRate","Epoch","Iteration","ExecutionEnvironment"];
            updateInfo(monitor,LearningRate=LearnRate, ...
                Epoch=epoch,Iteration=iteration, ExecutionEnvironment ="GPU");
            monitor.Progress = 100 * iteration/numIterations;

            % Update the plots of network scores and loss
            if mod(iteration, 100) == 0
                addpoints(lineScoreGenXToY, iteration, double(gather(extractdata(scores{1}))));
                addpoints(lineScoreDiscY, iteration, double(gather(extractdata(scores{2}))));
                drawnow limitrate;
                legend(scoreAxesY, 'Gen','Disc');
                % Update the Validation title with training progress information.
                T = duration(0,0,toc(start),'Format','hh:mm:ss');
                title( "Ep: " + epoch + ", " +  "Iter: " + iteration + ", " + ...
                    "SSIM: " + ValLoss + "," + "Elap: " + string(T));
            end
        end
        % ---------------------------------------------------------------
        % Decay learning rate once per epoch
        % ---------------------------------------------------------------
        if mod(epoch,decayEveryEpochs) == 0
            LearnRate = LearnRate * decayFactor;
            fprintf('Epoch %d: learning rate decayed to %g\n', ...
                epoch, LearnRate);
        end

        % Save model every n epochs
        if mod(epoch,100) == 0
            fileName = model_name + epoch + ".mat";
            save(fullfile(checkpointDir, fileName), ...
                'genX2Y', 'disc1', 'disc2');
        end
    end

    % Save Final Model
    fileName = model_name + epoch + ".mat";
    save(fullfile(checkpointDir, fileName), ...
        'genX2Y', 'disc1', "disc2");
end

%% Suport functions
% MatRead
function data = matReadImg(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data = data(:,:,:,1);
end

function data = matReadSeg(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data = data(:,:,:,2);
end

function data = matReadRef(filename,nn)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data = data(:,:,:,nn); %1-t1c, 2-T2, 3-T2F
end

function data = matReadValImg(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end


% Model gradients for Conditional (Pix2Pix) 3D GAN
function [gradParamG, gradParamDisc1, gradParamDisc2, LossG, scores] = modelGradients...
   (input, realImage, segMask, generator, disc1, disc2, targetModality)

% ----------------------------------------------------------
%  FORWARD PASS 
% ----------------------------------------------------------
genImage = forward(generator, input);
% ----------------------------------------------------------
%  BALANCED LOSS WEIGHTS
% ----------------------------------------------------------
lambdaDiscriminator   = 0.5;
lambdaGenerator       = 3.0;
lambdaFeatureMatching = 5.0;
lambda_SSIM           = 2.0;
% ----------------------------------------------------------
%  CONDITIONAL DISCRIMINATOR & ADVERSARIAL LOSS
% ----------------------------------------------------------
[LossD1, GANLossG1, realPredFtrs1, genPredFtrs1, GScore1, DScore1] = ...
    pix2pixHDAdverserialLoss(input, realImage, genImage, disc1);

input2 = dlresize(input, Scale=0.5, Method="linear");

realImage2 = dlresize(realImage, Scale=0.5, Method="linear");

genImage2 = dlresize(genImage, Scale=0.5, Method="linear");

[LossD2, GANLossG2, realPredFtrs2, genPredFtrs2, GScore2, DScore2] = ...
    pix2pixHDAdverserialLoss(input2, realImage2, genImage2, disc2);

LossD = lambdaDiscriminator * (LossD1 + LossD2) * 0.5;

GANLossG = lambdaGenerator * (GANLossG1 + GANLossG2) * 0.5;

% ----------------------------------------------------------
% FEATURE MATCHING LOSS
% ----------------------------------------------------------
FMLoss1 = pix2pixHDFeatureMatchingLoss(realPredFtrs1, genPredFtrs1);
FMLoss2 = pix2pixHDFeatureMatchingLoss(realPredFtrs2, genPredFtrs2);
FMLoss = lambdaFeatureMatching * 0.5 * (FMLoss1 + FMLoss2);

% ----------------------------------------------------------
% MULTI-LABEL TUMOR-WEIGHTED L1 LOSS
% ----------------------------------------------------------
segMask_numeric = extractdata(gather(segMask));

net_core   = (segMask_numeric == 1); % Necrotic/non-enhancing core
edema_mask = (segMask_numeric == 2); % Peritumoral edema
enhancing  = (segMask_numeric == 4); % Enhancing tumor core

w_background = 1.0;
switch lower(targetModality)
    case 't1c'
        w_edema     = 1.0;
        w_core      = 2.0; %4
        w_enhancing = 5.0; % 8% High weight for contrast enhancement
    case {'t2', 't2f'}
        w_edema     = 5.0; % High weight for large peritumoral edema
        w_core      = 2.0;
        w_enhancing = 1.0; % Low/negligible weight on T2/FLAIR
    otherwise
        error('Unsupported target modality (ModelGrad): %s', ...
            char(targetModality));
end

multiLabelWeightMap = ones(size(segMask_numeric), 'single') * w_background;

multiLabelWeightMap(edema_mask) = w_edema;
multiLabelWeightMap(net_core) = w_core;
multiLabelWeightMap(enhancing) = w_enhancing;

% Normalize weights
multiLabelWeightMap = multiLabelWeightMap ./ ...
                      mean(multiLabelWeightMap, 'all');

multiLabelWeightMap = dlarray(multiLabelWeightMap, 'SSSCB');

TumorWL1Loss = mean(abs(realImage - genImage) .* ...
                    multiLabelWeightMap, 'all');
% ----------------------------------------------------------
% SSIM LOSS COMPONENT
% ----------------------------------------------------------
SSIM_Loss = lambda_SSIM * customSSIM3D(realImage, genImage);

% ----------------------------------------------------------
% GENERATOR TOTAL LOSS
% ----------------------------------------------------------
LossG = GANLossG + FMLoss + ...
        TumorWL1Loss + SSIM_Loss;

% ----------------------------------------------------------
% GENERATOR GRADIENTS
% ----------------------------------------------------------
gradParamG = dlgradient(LossG, generator.Learnables, RetainData=true);

% ----------------------------------------------------------
% DISCRIMINATOR GRADIENTS
% ----------------------------------------------------------
gradParamDisc1 = dlgradient(LossD, disc1.Learnables);
gradParamDisc2 = dlgradient(LossD, disc2.Learnables);

% ----------------------------------------------------------
% OUTPUT LOGGING
% ----------------------------------------------------------
LossG = gather(extractdata(LossG));

scores = {(GScore1 + GScore2) * 0.5, ...
    (DScore1 + DScore2) * 0.5};
end


% ----------------------------------------------------------
% CONDITIONAL ADVERSARIAL LOSS FUNCTION
% ----------------------------------------------------------
function [DLoss, GLoss, realPredFtrsD, genPredFtrsD, GScore, DScore] = ...
    pix2pixHDAdverserialLoss(srcIm, realIm, genIm, discriminator)

featureNames = ["act_C64", ...
                "act_C128", ...
                "act_C256", ...
                "act_C512", ...               
                "conv_final"];

% Concatenate source MRI with targets along Channel Dim (Dim 4)
realPair = cat(4, srcIm, realIm); 
genPair  = cat(4, srcIm, genIm);

% Combine real and generated pairs along Batch Dim (Dim 5) for single pass
combinedPairs = cat(5, realPair, genPair);

combinedPredFtrs = cell(size(featureNames));
[combinedPredFtrs{:}] = forward(discriminator, combinedPairs, Outputs=featureNames);

batchSize = size(realIm, 5);

% Split discriminator features
realPredFtrsD = cell(size(featureNames));
genPredFtrsD  = cell(size(featureNames));

for k = 1:numel(featureNames)
    realPredFtrsD{k} = combinedPredFtrs{k}(:,:,:,:,1:batchSize);
    genPredFtrsD{k}  = combinedPredFtrs{k}(:,:,:,:,batchSize+1:end);
end

realPred = realPredFtrsD{end};
fakePred = genPredFtrsD{end};

% Logging scores using sigmoid on logits
GScore = mean(sigmoid(fakePred), "all");
DScore = 0.5 * mean(sigmoid(realPred), "all") + ...
         0.5 * mean(1 - sigmoid(fakePred), "all");

% % LSGAN Mean Squared Error Losses
% DLoss = mean((1 - realPred).^2 + fakePred.^2, "all");
% GLoss = mean((1 - fakePred).^2, "all");
% replacing with Hinge Loss
% Generator Hinge Loss
GLoss = -mean(fakePred, "all");
% Discriminator Hinge Loss
DLoss = mean(max(0, 1 - realPred), "all") + mean(max(0, 1 + fakePred), "all");
end

% ----------------------------------------------------------
% FEATURE MATCHING LOSS FUNCTION
% ----------------------------------------------------------
function fmLoss = pix2pixHDFeatureMatchingLoss(realF, fakeF)
fmLoss = 0;
for i = 1:numel(realF) - 1
    realFeat = realF{i};
    fakeFeat = fakeF{i};
    featureScale = mean(abs(realFeat), 'all') + 1e-4;
    fmLoss = fmLoss + ...
        mean(abs(realFeat - fakeFeat), 'all') ./ featureScale;
end
end

function lossSSIM = customSSIM3D(im1, im2)
    % Custom differentiable 3D SSIM loss for dlarray tensors [SSSCB]
    % Dynamic range L = 2 (for image intensities in range [-1, 1])
    C1 = (0.01 * 2)^2;
    C2 = (0.03 * 2)^2;

    % Create a 3D Gaussian kernel (size 3x3x3)
    k1D = single([0.25, 0.5, 0.25]);
    kernel3D = reshape(k1D, [3 1 1]) .* reshape(k1D, [1 3 1]) .* reshape(k1D, [1 1 3]);
    kernel3D = reshape(kernel3D, [3 3 3 1 1]); % [H W D In Out]
    
    if canUseGPU
        kernel3D = gpuArray(kernel3D);
    end

    % 3D convolution helper for spatial smoothing per channel/batch
    smooth3D = @(x) dlconv(x, kernel3D, 0, "Padding", "same");

    % Means
    mu1 = smooth3D(im1);
    mu2 = smooth3D(im2);

    mu1_sq = mu1 .^ 2;
    mu2_sq = mu2 .^ 2;
    mu1_mu2 = mu1 .* mu2;

    % Variances and Covariance
    sigma1_sq = smooth3D(im1 .^ 2) - mu1_sq;
    sigma2_sq = smooth3D(im2 .^ 2) - mu2_sq;
    sigma12   = smooth3D(im1 .* im2) - mu1_mu2;

    % SSIM Map
    num = (2 * mu1_mu2 + C1) .* (2 * sigma12 + C2);
    den = (mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2);
    ssimMap = num ./ (den + 1e-6);

    % Loss = 1 - mean(SSIM)
    lossSSIM = 1 - mean(ssimMap, 'all');
end
%% Initilialize figure
function [figureHandle, tileHandle, imageAxes1,imageAxes2,imageAxes3, scoreAxesY, LossAxesY, ...
    lineScoreGenXToY, lineScoreDiscY, lineValLossX2Y] = initializeTrainingPlot_SupGAN()
% Create a wide figure for displaying training progress
figureHandle = figure('Name','Validation Progress');
figureHandle.Position(3) = ...
    1.5 * figureHandle.Position(3); % Make figure wider
% Layout: 2 rows x 2 columns
tileHandle = tiledlayout(figureHandle, 2, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');
% Tile 1: For image display (3x2 view will be inserted here)
imageAxes1 = nexttile(tileHandle, 1);
title(imageAxes1, 'Validation Slices');
axis off;
% Tile 2: For image display (3x2 view will be inserted here)
imageAxes2 = nexttile(tileHandle, 2);
title(imageAxes2, 'Validation Slices');
axis off;
% Tile 3: For image display (3x2 view will be inserted here)
imageAxes3 = nexttile(tileHandle, 3);
title(imageAxes3, 'Validation Slices');
axis off;
% Tile 4: Validation loss
LossAxesY = nexttile(tileHandle, 4);
xlabel(LossAxesY, "Iteration");
ylabel(LossAxesY, "Loss");
grid(LossAxesY, "on");
% Tile 3: Generator & Discriminator scores
scoreAxesY = nexttile(tileHandle, 6);
xlabel(scoreAxesY, "Iteration");
ylabel(scoreAxesY, "Score");
grid(scoreAxesY, "on");
% Initialize animated lines
lineScoreGenXToY = animatedline(scoreAxesY, ...
    'Color', 'r', ...
    'LineWidth', 1.5);
lineScoreDiscY = animatedline(scoreAxesY, ...
    'Color', 'g', ...
    'LineWidth', 1.5);
lineValLossX2Y = animatedline(LossAxesY, ...
    'Color', 'b', ...
    'Marker', '*', ...
    'LineWidth', 1.2);
end
%% VALIDATION IMAGES
function [genImage, ssim_val] = displayGeneratedValImages ...
    (dsVal, imageAxes1, imageAxes2, imageAxes3, ...
    LossAxesY, lineValLossX2Y, genXToY, iteration, patchSize)

% Read mini-batch of data
[imX, imY] = next(dsVal);

 % convert to single for faster trian - GPU
seg = single(imX(:,:,:,2,:)); seg(seg > 0) = 1;
imX = single(imX(:,:,:,1,:)); 
imY = single(imY); 


% convert to dlarray
imX = dlarray(imX,'SSSCB');
imY = dlarray(imY,'SSSCB');

% ----------------------------------------------------------
% Generate images using the generator
% ----------------------------------------------------------
genImage = GenerateFullDatafromPatch( ...
    imX, genXToY, patchSize);

% Extract image data
inpImage = extractdata(gather(imX));
realImage = extractdata(gather(imY));
seg = extractdata(gather(seg));

% Convert to 3-D volumes for visualization
inpImage = squeeze(inpImage);
realImage = squeeze(realImage);
genImage = squeeze(genImage);


% Find mid slices based on segmentation mass
sumOverYZ = squeeze(sum(sum(seg, 3), 2));
midX = find(sumOverYZ == max(sumOverYZ), 1, 'first');

sumOverXZ = squeeze(sum(sum(seg, 3), 1));
midY = find(sumOverXZ == max(sumOverXZ), 1, 'first');

sumOverXY = squeeze(sum(sum(seg, 2), 1));
midZ = find(sumOverXY == max(sumOverXY), 1, 'first');

% Axial view
axialInp = inpImage(:,:,midZ);
axialReal = realImage(:,:,midZ);
axialGenerated = genImage(:,:,midZ);

% Sagittal view
sagittalInp = ...
    flipud(permute(squeeze(inpImage(midX,:,:)),[2 1]));

sagittalReal = ...
    flipud(permute(squeeze(realImage(midX,:,:)),[2 1]));

sagittalGenerated = ...
    flipud(permute(squeeze(genImage(midX,:,:)),[2 1]));

% Coronal view
coronalInp = ...
    flipud(permute(squeeze(inpImage(:,midY,:)),[2 1]));

coronalReal = ...
    flipud(permute(squeeze(realImage(:,midY,:)),[2 1]));

coronalGenerated = ...
    flipud(permute(squeeze(genImage(:,midY,:)),[2 1]));

% ---- Plot axial ----
cla(imageAxes1);

imageResults = cat(2, ...
    axialInp, ...
    axialReal, ...
    axialGenerated);

imshow(imageResults,[-1,1],...
    'Parent',imageAxes1);

axis(imageAxes1,'image');

title(imageAxes1,...
    "Axial: T1 (Left) Vs Real (Mid) vs Generated (Right)");

% ---- Plot sagittal ----
cla(imageAxes2);

imageResults = cat(2, ...
    sagittalInp, ...
    sagittalReal, ...
    sagittalGenerated);

imshow(imageResults,[-1,1],...
    'Parent',imageAxes2);

axis(imageAxes2,'image');

title(imageAxes2,...
    "Sagittal: T1 (Left) Vs Real (Mid) vs Generated (Right)");

% ---- Plot coronal ----
cla(imageAxes3);

imageResults = cat(2, ...
    coronalInp, ...
    coronalReal, ...
    coronalGenerated);

imshow(imageResults,[-1,1],...
    'Parent',imageAxes3);

axis(imageAxes3,'image');

title(imageAxes3,...
    "Coronal: T1 (Left) Vs Real (Mid) vs Generated (Right)");

% Compute validation SSIM
ssim_val = multissim3(realImage,genImage);

% Update the validation loss plot
addpoints(lineValLossX2Y,...
    iteration,...
    double(1-ssim_val));
legend(LossAxesY, 'ValLoss (1-SSIM)');
drawnow limitrate;
end


%% Discriminator
function net = PatchGAN3DDiscriminator(inputSize, numChannels)
% PATCHGAN3DDISCRIMINATOR 3D PatchGAN Discriminator for Image Translation.
%
% Inputs:
%   inputSize   - Spatial dimensions [H, W, D] (e.g., [128 128 128])
%   numChannels - Total input channels (e.g., 2 for concatenated [Source, Target])

if nargin < 2
    % Default to paired input: 1 channel Source + 1 channel Target/Generated
    numChannels = 2; 
end

fullInputSize = [inputSize, numChannels];
net = layerGraph();

% -------------------------------------------------------------
% Layer Architecture Definitions
% -------------------------------------------------------------
layers = [
    % Input Layer (Accepts concatenated source + target volumes)
    image3dInputLayer(fullInputSize, "Name", "input_top", "Normalization", "none")

    % C64: No normalization on the first layer (Standard PatchGAN)
    convolution3dLayer(4, 64, "Stride", 2, "Padding", "same", ...
        "Name", "conv_C64", "WeightsInitializer", "narrow-normal")
    leakyReluLayer(0.2, "Name", "act_C64")

    % C128
    convolution3dLayer(4, 128, "Stride", 2, "Padding", "same", ...
        "Name", "conv_C128", "WeightsInitializer", "narrow-normal")
    groupNormalizationLayer(128, "Name", "norm_C128") % Instance Norm (groups = channels)
    leakyReluLayer(0.2, "Name", "act_C128")

    % C256
    convolution3dLayer(4, 256, "Stride", 2, "Padding", "same", ...
        "Name", "conv_C256", "WeightsInitializer", "narrow-normal")
    groupNormalizationLayer(256, "Name", "norm_C256")
    leakyReluLayer(0.2, "Name", "act_C256")

    % C512 (Stride 1 to maintain patch spatial resolution)
    convolution3dLayer(4, 512, "Stride", 1, "Padding", "same", ...
        "Name", "conv_C512", "WeightsInitializer", "narrow-normal")
    groupNormalizationLayer(512, "Name", "norm_C512")
    leakyReluLayer(0.2, "Name", "act_C512")

    % Final 1-channel PatchGAN Output
    convolution3dLayer(4, 1, "Stride", 1, "Padding", "same", ...
        "Name", "conv_final", "WeightsInitializer", "narrow-normal")
];
net = addLayers(net, layers);
net = dlnetwork(net);
end

%% Generate full image data from patch (using Raised-Cosine / Hann Window)
function reconstructed_volume =   GenerateFullDatafromPatch...
  (inputVolume, net, patchSize)

% -------------------------------------------------------------
% Convert input to numeric data
% -------------------------------------------------------------
if isa(inputVolume,'dlarray')
    inputVolume = extractdata(inputVolume);
end

inputVolume = gather(inputVolume);

% -------------------------------------------------------------
% Remove channel and batch dimensions
% -------------------------------------------------------------
if ndims(inputVolume) == 5
    inputVolume = inputVolume(:,:,:,:,1);
end

inputVolume = squeeze(inputVolume);

volSize = size(inputVolume);
volSize = volSize(1:3);

% -------------------------------------------------------------
% Volume size & Padding
% -------------------------------------------------------------
stride = floor(patchSize/2);
pad = floor(patchSize/2);

% Pad explicitly with -1 (background value)
volPad = padarray(single(inputVolume),pad,-1,'both');

padSize = size(volPad);
padSize = padSize(1:3);

% -------------------------------------------------------------
% Output buffers
% -------------------------------------------------------------
reconstructed = zeros(padSize,'single');
weightSum = zeros(padSize,'single');

% -------------------------------------------------------------
% 3D Raised-Cosine (Hann) Weighting Window
% -------------------------------------------------------------
wx = hann(patchSize(1), 'periodic');
wy = hann(patchSize(2), 'periodic');
wz = hann(patchSize(3), 'periodic');

W = reshape(single(wx),[],1,1) .* ...
    reshape(single(wy),1,[],1) .* ...
    reshape(single(wz),1,1,[]);

% Add a small constant to ensure non-zero weights at the absolute boundaries
W = W + 1e-4;

% -------------------------------------------------------------
% Move Raised-Cosine weighting window to GPU
% -------------------------------------------------------------
if canUseGPU
    Wgpu = gpuArray(W);
else
    Wgpu = W;
end

% -------------------------------------------------------------
% Patch locations
% -------------------------------------------------------------
xs = 1:stride(1):padSize(1)-patchSize(1)+1;
ys = 1:stride(2):padSize(2)-patchSize(2)+1;
zs = 1:stride(3):padSize(3)-patchSize(3)+1;

if xs(end) ~= padSize(1)-patchSize(1)+1
    xs(end+1) = padSize(1)-patchSize(1)+1;
end

if ys(end) ~= padSize(2)-patchSize(2)+1
    ys(end+1) = padSize(2)-patchSize(2)+1;
end

if zs(end) ~= padSize(3)-patchSize(3)+1
    zs(end+1) = padSize(3)-patchSize(3)+1;
end

% -------------------------------------------------------------
% Sliding-window inference
% -------------------------------------------------------------
for z = zs
    for y = ys
        for x = xs

            patch = volPad(...
                x:x+patchSize(1)-1,...
                y:y+patchSize(2)-1,...
                z:z+patchSize(3)-1);

            % Add channel and batch dimensions
            patch = reshape(single(patch), [patchSize 1 1]);
            patch = dlarray(patch,'SSSCB');

            if canUseGPU
                patch = gpuArray(patch);
            end

            % Prediction
            pred = predict(net, patch);
            pred = gather(extractdata(pred));
            pred = squeeze(pred);

            % Weighted accumulation
            reconstructed(x:x+patchSize(1)-1,...
                y:y+patchSize(2)-1,...
                z:z+patchSize(3)-1) = reconstructed(...
                x:x+patchSize(1)-1,...
                y:y+patchSize(2)-1,...
                z:z+patchSize(3)-1) + pred .* W;

            weightSum(x:x+patchSize(1)-1,...
                y:y+patchSize(2)-1,...
                z:z+patchSize(3)-1) = weightSum(...
                x:x+patchSize(1)-1,...
                y:y+patchSize(2)-1,...
                z:z+patchSize(3)-1) + W;
        end
    end
end

% -------------------------------------------------------------
% Normalize
% -------------------------------------------------------------
weightSum(weightSum == 0) = 1;
reconstructed = reconstructed ./ weightSum;

% -------------------------------------------------------------
% Remove padding
% -------------------------------------------------------------
reconstructed_volume = reconstructed(...
    pad(1)+1:pad(1)+volSize(1),...
    pad(2)+1:pad(2)+volSize(2),...
    pad(3)+1:pad(3)+volSize(3));

% -------------------------------------------------------------
% Background Masking for [-1, +1] range
% -------------------------------------------------------------
brainMask = inputVolume > -0.98;
reconstructed_volume(~brainMask) = -1.0;

% Final output clamp
reconstructed_volume = max(min(reconstructed_volume,1),-1);

end

%% Extract Patches around tumor
function [batchInp, batchRef, batchSeg] = extractTumorGuidedPatches(ramInp, ramRef, ramSeg, patchSize, miniBatchSize)
batchInp = zeros([patchSize 1 miniBatchSize], 'single');
batchRef = zeros([patchSize 1 miniBatchSize], 'single');
batchSeg = zeros([patchSize 1 miniBatchSize], 'single');

numVolumes = numel(ramInp);
halfP = floor(patchSize / 2);
numTumorInBatch = round(0.6 * miniBatchSize); % n% tumor region

for b = 1:miniBatchSize
    volIdx = randi(numVolumes);
    inpVol = ramInp{volIdx};
    refVol = ramRef{volIdx};
    segVol = ramSeg{volIdx};

    volSize = size(inpVol);

    xRange = (halfP(1) + 1):(volSize(1) - halfP(1));
    yRange = (halfP(2) + 1):(volSize(2) - halfP(2));
    zRange = (halfP(3) + 1):(volSize(3) - halfP(3));

    validBoundaryMask = false(volSize);
    validBoundaryMask(xRange, yRange, zRange) = true;

    tumorMask = (segVol > 0);
    isTumorPatch = (b <= numTumorInBatch);

    if isTumorPatch
        candidateCoords = find(tumorMask & validBoundaryMask);
        if isempty(candidateCoords)
            candidateCoords = find(~tumorMask & validBoundaryMask);
        end
    else
        candidateCoords = find(~tumorMask & validBoundaryMask);
    end

    selectedLinearIdx = candidateCoords(randi(numel(candidateCoords)));
    [cx, cy, cz] = ind2sub(volSize, selectedLinearIdx);

    xIdx = (cx - halfP(1)):(cx + halfP(1) - 1);
    yIdx = (cy - halfP(2)):(cy + halfP(2) - 1);
    zIdx = (cz - halfP(3)):(cz + halfP(3) - 1);

    batchInp(:,:,:,1,b) = inpVol(xIdx, yIdx, zIdx);
    batchRef(:,:,:,1,b) = refVol(xIdx, yIdx, zIdx);
    batchSeg(:,:,:,1,b) = segVol(xIdx, yIdx, zIdx);
end
end

function [ramInp, ramRef, ramSeg] = refreshVolumeBuffer(imdsInp, imdsRef, imdsSeg, bufferSize, numVolumesTotal, nn)
selectedIndices = randperm(numVolumesTotal, bufferSize);

inpFiles = imdsInp.Files(selectedIndices);
refFiles = imdsRef.Files(selectedIndices);
segFiles = imdsSeg.Files(selectedIndices);

ramInp = cell(bufferSize, 1);
ramRef = cell(bufferSize, 1);
ramSeg = cell(bufferSize, 1);

parfor v = 1:bufferSize
    ramInp{v} = matReadImg(inpFiles{v});
    ramRef{v} = matReadRef(refFiles{v},nn);
    ramSeg{v} = matReadSeg(segFiles{v});
end
end



