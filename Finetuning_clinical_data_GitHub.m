% =========================================================================
% MATLAB Code for Research Publication
%
% Paper:
% "Multi-Contrast Brain MRI Synthesis for Radiotherapy Using a Dual-Scale Generative Adversarial Network with Volumetric Context"
%
% Author:
% SA Yoganathan
% Medical Physicist
% Saint John Regional Hospital
% Horizon Health Network
% Saint John, New Brunswick, Canada
%
% Contact:
% sa.yoganathan@horizonnb.ca
%
% License:
% This code is released for academic and research use only.
%
% Disclaimer:
% This software is provided "as is" without warranty of any kind.
% It is not intended for clinical use and has not been approved for
% medical diagnosis or treatment planning.
%
% Copyright (c) 2026 SA Yoganathan
% =========================================================================

% training data
D = 'Y:\Syn_MRI\Clinical_MR_Data\Train\Input';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
imdsInp = @(x) matReadX(x);
imdsInp = imageDatastore(F,'FileExtensions','.mat','ReadFcn',imdsInp);

D = 'Y:\Syn_MRI\Clinical_MR_Data\Train\Response';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
imdsRef = @(x) matReadY(x);
imdsRef = imageDatastore(F,'FileExtensions','.mat','ReadFcn',imdsRef);

% Validation data
D = 'Y:\Syn_MRI\Clinical_MR_Data\Valid\Input';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
Funct = @(x) matReadX(x);
imdsValInp = imageDatastore(F,'FileExtensions','.mat','ReadFcn',Funct);

D = 'Y:\Syn_MRI\Clinical_MR_Data\Valid\Response';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
Funct = @(x) matReadY(x);
imdsValRef = imageDatastore(F,'FileExtensions','.mat','ReadFcn',Funct);

% Extract batch
miniBatchSize = 10;
% Combine input and response datastores
dsTrain = combine(imdsInp, imdsRef);
dsVal = combine(imdsValInp, imdsValRef);

% Create minibatchqueue from the combined datastore - Test data
mbqTrain = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@(x,y) deal(cat(5,x{:}), cat(5,y{:})),...
    PartialMiniBatch="discard",...
    MiniBatchFormat=["SSSCB","SSSCB"],...
    OutputEnvironment="gpu");

% Create minibatchqueue from the combined datastore - Validation data
mbqVal = minibatchqueue(dsVal,...
    MiniBatchSize=1,...
    MiniBatchFcn=@(x,y) deal(cat(5,x{:}), cat(5,y{:})),...
    PartialMiniBatch="discard",...
    MiniBatchFormat=["SSSCB","SSSCB"],...
    OutputEnvironment="gpu");
%% Model
Nslices = 32; % number of slices per input block
pad = Nslices/2;   % symmetric padding → Nslices/2 slices
inputSize = [240 240 Nslices];

% import BraTs - trained Model(s)
net = importdata('Y:\Syn_MRI\Models\synMRI-epoch12_2.5D_BraT_T2_newupdated.mat');
genX2Y = net.genX2Y; disc2D1 = net.disc2D1; disc2D2 = net.disc2D2;
disc3D1 = net.disc3D1; disc3D2 = net.disc3D2;

%% Load VGG Net for 2D perceptual loss
netVGGTF = imagePretrainedNetwork("vgg19");
netVGGTF = dlnetwork(netVGGTF.Layers(1:38));
% Convert to layer graph (needed for replaceLayer)
lgraph = layerGraph(netVGGTF);
% Replace the input layer with new size
inp = imageInputLayer([inputSize(1), inputSize(2), 3], ...
    "Normalization", "none", "Name", "unet_input");
lgraph = replaceLayer(lgraph, "input", inp);
% Convert to dlnetwork
netVGG = dlnetwork(lgraph);

clear net lgraph netVGGTF
%% Training
numEpochs = 25; 
trailAvgGen = []; trailAvgSqGen = [];
trailAvgDiscSc1 = []; trailAvgSqDiscSc1 = [];
trailAvgDiscSc2 = []; trailAvgSqDiscSc2 = [];
gradDecayFac = 0.5; sqGradDecayFac = 0.999;

trailAvgDisc3D1 = []; trailAvgSqDisc3D1 = [];
trailAvgDisc3D2 = []; trailAvgSqDisc3D2 = [];

%Training Model
ValLoss=2; validFreq = 50;
epoch = 0; start = tic;
iteration2D = 0; iteration3D = 0;

% learning Rate parameters
initialLearnRate = 0.0005;
LearnRate = initialLearnRate;
decayFactor = 0.5;       % Reduce LR by N% every decay step
decayEveryEpochs = 2;   % Decay every N epochs
updateLR = decayEveryEpochs;

alpha = 5;   % weight for L1 loss
beta  = 10;   % weight for SSIM loss


% monitor trainging progress
monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss");
monitor.Info=["LearningRate","Epoch","Iteration","ExecutionEnvironment"];
monitor.XLabel = "Iteration";
monitor.Progress = 0;

sliceReduction = floor(150/1);
numObservationsTrain = numel(dsTrain.UnderlyingDatastores{1,1}.Files);
numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize)*sliceReduction;
numIterations = numEpochs*numIterationsPerEpoch;

dataDir = fullfile('Y:\Syn_MRI\');
% Create a directory to store checkpoints
checkpointDir = fullfile(dataDir,"Models");
if ~exist(checkpointDir,"dir")
    mkdir(checkpointDir);
end

% Initialize plots for training progress
[figureHandle, tileHandle, imageAxes1, imageAxes2, imageAxes3, scoreAxesY, LossAxesY, ...
    lineScoreGenXToY, lineScoreDiscY, lineValLossX2Y] = initializeTrainingPlot();
% Make sure figure is visible (especially in Live Script)
set(figureHandle, 'Visible', 'on');
executionEnvironment = "gpu";

% Training loop
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    % Reset and shuffle the data
    reset(mbqTrain); shuffle(mbqTrain);

    % Loop over mini-batches
    while hasdata(mbqTrain) && ~monitor.Stop
        % Read mini-batch of data
        [InpT1Seg, OutPut] = next(mbqTrain);

        % ----------------- calculate 2.5D loss ---------------------------
        iteration3D = iteration3D + 1;
        N = min(10, round(miniBatchSize/2));  % number of random samples

        % Pre-allocate
        predCell = cell(1,N); trueCell = cell(1,N); segCell  = cell(1,N);

        % Predict ONE volume at a time
        for ii = 1:N
            bb = randi([1 size(OutPut,5)],1);
            inpVol  = InpT1Seg(:,:,:,1,bb);
            trueVol = OutPut(:,:,:,:,bb);
            seg     = InpT1Seg(:,:,:,2,bb) > 0;
            predVol = GenerateFullData(inpVol, genX2Y, Nslices);   % dlarray
            predCell{ii} = predVol;
            trueCell{ii} = trueVol;
            segCell{ii}  = seg;
        end
        % Concatenate dlarrays
        dlYPredBatch = cat(5,predCell{:});
        dlYTrueBatch = cat(5,trueCell{:});
        segBatch     = cat(5,segCell{:});
        dlYPredBatch = dlarray(dlYPredBatch,"SSSCB");
        dlYTrueBatch = dlarray(dlYTrueBatch,"SSSCB");
        segBatch     = dlarray(segBatch,"SSSCB");

        % Masked L1 loss       
        diff = abs(dlYTrueBatch - dlYPredBatch);   % H W D 1 N
        diff(~segBatch) = 0;
        perSampleNum = squeeze(sum(diff,[1 2 3]));          % N×1
        nnz_per_sample = squeeze(sum(segBatch,[1 2 3]));    % N×1
        nnz_per_sample(nnz_per_sample==0) = 1;              % avoid div/0

        % Compute mean abs of true within mask
        true_abs = abs(dlYTrueBatch);
        true_abs(~segBatch) = 0;
        perSampleDen = squeeze(sum(true_abs,[1 2 3]));
        meanAbsVal = perSampleDen ./ nnz_per_sample;
        meanAbsVal(meanAbsVal==0) = 1e-6;

        L1_Loss = perSampleNum ./ nnz_per_sample;
        L1_Loss = L1_Loss ./ meanAbsVal;
        L1_Loss = mean(L1_Loss);   % final aggregated

        %  Adversarial loss
        [D3DLoss, G3DLoss, grad3DScale1, grad3DScale2] = dlfeval( ...
            @disc3DLossAndGradients, disc3D1, disc3D2, segBatch, dlYTrueBatch, dlYPredBatch);

       % Update discriminator 3D scale1
       [disc3D1, trailAvgDisc3D1, trailAvgSqDisc3D1] = adamupdate( ...
           disc3D1, grad3DScale1, trailAvgDisc3D1, trailAvgSqDisc3D1, iteration3D, ...
           LearnRate, gradDecayFac, sqGradDecayFac);

       % Update discriminator 3D scale2
       [disc3D2, trailAvgDisc3D2, trailAvgSqDisc3D2] = adamupdate( ...
           disc3D2, grad3DScale2, trailAvgDisc3D2, trailAvgSqDisc3D2, iteration3D, ...
           LearnRate, gradDecayFac, sqGradDecayFac);

        % Final loss - 2.5D       
        final3dLoss = alpha * L1_Loss + beta * G3DLoss;
        clear dlYPredBatch dlYTrueBatch segBatch inpVol predVol seg diff true_abs


        % --------------------------- 2D Loss --------------------------
        % read data
        Inp = single(InpT1Seg(:,:,:,1,:));
        OutPut = single(OutPut); seg = InpT1Seg(:,:,:,2,:);
        [H, W, D, C, B] = size(Inp);

        % Random permutation of slice indices
        randIdx = randperm(D,sliceReduction);

        % Loop over slices in random order
        for i = 1:numel(randIdx)
            z = randIdx(i);   % pick slice index randomly, no repetition
            iteration2D = iteration2D + 1;
            zstart = z - pad; zend   = z + pad - 1;
            if zstart < 1
                % Need pre-padding
                numPre = 1 - zstart;                 % how many slices missing before 1
                valid  = Inp(:,:,1:zend,:,:);            % available slices
                dlX    = cat(3, zeros(H,W,numPre, C, B,'like',Inp), valid);
            elseif zend > D
                % Need post-padding
                numPost = zend - D;                  % how many slices missing after D
                valid   = Inp(:,:,zstart:D,:,:);         % available slices
                dlX     = cat(3, valid, zeros(H,W,numPost, C, B,'like',Inp));
            else
                % Normal case (fully inside volume)
                dlX = Inp(:,:,zstart:zend,:,:);
            end

            % Merge slice and channel dims → [H, W, Nslices*C, B]
            dlX = reshape(dlX, H, W, [], B);

            % Sanity check: dlX should always have Nslices in 3rd dim
            assert(size(dlX,3) == Nslices);

            % Convert to dlarray (H,W,Channels,Batch) where Channels=Nslices
            dlX = dlarray(single(dlX), "SSCB");

            % Ground truth: the actual z-th slice (predicted slice corresponds to centerIdx)
            dlY = OutPut(:,:,z,:,:);
            dlY = reshape(dlY, H, W, [], B);
            dlY = dlarray(single(dlY), "SSCB");

            dlSeg = seg(:,:,z,:,:);
            dlSeg = reshape(dlSeg, H, W, [], B);
            dlSeg = dlarray(single(dlSeg), "SSCB");

            % Calculate the loss and gradients
            [gradParamsG, gradParamsDScale1, gradParamsDScale2, lossGGAN, lossGFM, lossGVGG, lossD, scores] = ...
                dlfeval(@modelGradients, dlX, dlY, genX2Y, disc2D1, disc2D2, netVGG, final3dLoss);

            % Update parameters of generator
            [genX2Y,trailAvgGen, trailAvgSqGen] = adamupdate( ...
                genX2Y, gradParamsG, trailAvgGen, trailAvgSqGen, iteration2D, ...
                LearnRate, gradDecayFac, sqGradDecayFac);

            % Update parameters of discriminator scale1
            [disc2D1, trailAvgDiscSc1, trailAvgSqDiscSc1] = adamupdate( ...
                disc2D1, gradParamsDScale1, trailAvgDiscSc1, trailAvgSqDiscSc1, iteration2D, ...
                LearnRate, gradDecayFac, sqGradDecayFac);

            % Update the discriminator scale2 parameters
            [disc2D2, trailAvgDiscSc2, trailAvgSqDiscSc2] = adamupdate( ...
                disc2D2, gradParamsDScale2, trailAvgDiscSc2, trailAvgSqDiscSc2, iteration2D, ...
                LearnRate, gradDecayFac, sqGradDecayFac);

            % Update the plots of network scores and loss
            addpoints(lineScoreGenXToY,iteration2D, double(gather(extractdata(scores{1}))));

            addpoints(lineScoreDiscY,iteration2D,double(gather(extractdata(scores{2}))));
            legend(scoreAxesY, 'Gen','Disc');

            drawnow;

            % Update the Validation title with training progress information.
            T = duration(0,0,toc(start),'Format','hh:mm:ss');
            title( "Ep: " + epoch + ", " +  "Iter: " + iteration2D + ", " + ...
                "SSIM: " + ValLoss + "," + "Elap: " + string(T));

            % Update the training progress - Record metrics.
            Loss = lossGGAN + lossGFM + lossGVGG + final3dLoss;

            % --- Update the monitor ---
            recordMetrics(monitor, iteration2D, ...
                TrainingLoss=Loss);

            monitor.Info=["LearningRate","Epoch","Iteration","ExecutionEnvironment"];

            updateInfo(monitor,LearningRate=LearnRate, ...
                Epoch=epoch, ...
                Iteration=iteration2D, ExecutionEnvironment ="GPU");

            monitor.Progress = 100 * iteration2D/numIterations;

            % Validation display
            if mod(iteration2D,validFreq) == 0 || iteration2D == 1
                % Reset Validation data
                if hasdata(mbqVal) == 0
                    reset(mbqVal); shuffle(mbqVal);
                end
                [~, ValLoss] = displayGeneratedValImages(mbqVal, imageAxes1, imageAxes2, imageAxes3, LossAxesY, lineValLossX2Y, genX2Y, iteration2D, Nslices);
                drawnow expose;     % Force GUI refresh
                pause(0.05);        % Ensure Live Script updates figure
                ValLoss = round(ValLoss,3);
            end

            % --- Decay learning rate every decayEveryEpochs ---
            if mod(epoch, updateLR) == 0
                LearnRate = LearnRate * decayFactor;
                fprintf('Epoch %d: learning rate decayed to %g\n', epoch, LearnRate);
                updateLR = updateLR + decayEveryEpochs;
            end
        end
        clear InpT1Seg Inp OutPut dlX dlY dlSeg
    end

    % Save model every 5 epochs
    if mod(epoch,5) == 0
        % filename with epoch number
        fileName = "synMRI-epoch" + epoch + ".mat";
        % save
        save(fullfile(checkpointDir, fileName), ...
            'genX2Y', 'disc2D1', 'disc2D2','disc3D1', 'disc3D2');
    end
end
% Save Final one
fileName = "synMRI-epoch" + epoch + ".mat";
save(fullfile(checkpointDir, fileName), ...
    'genX2Y', 'disc2D1', 'disc2D2','disc3D1', 'disc3D2');



%% 2.5D Loss & Gradients
function [D3DTotalLoss, G3DTotalLoss, grad3D1, grad3D2] = disc3DLossAndGradients( ...
        disc3D1, disc3D2, segBatch, YTrueVol, YPredVol)
    
    % Crop to tumor region    
    [YTrueVol, YPredVol] = cropToForeground3D(segBatch, YTrueVol, YPredVol);
   
    % Convert to dlarray
    YTrueVol = dlarray(single(YTrueVol), "SSSCB");
    YPredVol = dlarray(single(YPredVol), "SSSCB");
   
    % SCALE 1 (full resolution)   
    realPred1 = forward(disc3D1, YTrueVol);
    fakePred1 = forward(disc3D1, YPredVol);
    D3D1Loss = mean((1 - realPred1).^2 + fakePred1.^2, "all");
    G3D1Loss = mean((1 - fakePred1).^2, "all");
   
    % SCALE 2 (downsampled)  
    YTrueVol2 = dlresize(YTrueVol, Scale=0.5, Method="linear");
    YPredVol2 = dlresize(YPredVol, Scale=0.5, Method="linear");
    realPred2 = forward(disc3D2, YTrueVol2);
    fakePred2 = forward(disc3D2, YPredVol2);
    D3D2Loss = mean((1 - realPred2).^2 + fakePred2.^2, "all");
    G3D2Loss = mean((1 - fakePred2).^2, "all");

    % Total loss for Disc & Gen   
    D3DTotalLoss = 0.5 * (D3D1Loss + D3D2Loss);
    G3DTotalLoss = 0.5 * (G3D1Loss + G3D2Loss);
    
    % Compute gradients 
    grad3D1 = dlgradient(D3DTotalLoss, disc3D1.Learnables, RetainData=true);
    grad3D2 = dlgradient(D3DTotalLoss, disc3D2.Learnables);
end

%% 2D Loss & Gradients
function [gradParamsG,gradParamsDScale1,gradParamsDScale2,...
    lossGGAN,lossGFM,lossGVGG,lossD,scores] = ...
    modelGradients(input, realImage, generator, ...
    discScale1, discScale2, ...
    netVGG, full3dLoss)

% GENERATOR FORWAR
genImage = forward(generator, input);

% LOSS WEIGHTS
lambdaDisc = 1; lambdaGen = 1; lambdaFM = 5; lambdaVGG = 5;

% DISCRIMINATOR SCALE 1 (full resolution)
[DLossScale1, GLossScale1, realPred1, fakePred1, GScore1, DScore1] = ...
    pix2pixHDAdverserialLoss(realImage, genImage, discScale1);

% DISCRIMINATOR SCALE 2 — DOWNSAMPLE
resizedReal = dlresize(realImage,Scale=0.5,Method="linear");
resizedGen  = dlresize(genImage,Scale=0.5,Method="linear");
[DLossScale2, GLossScale2, realPred2, fakePred2, GScore2, DScore2] = ...
    pix2pixHDAdverserialLoss(resizedReal, resizedGen, discScale2);

% FEATURE MATCHING LOSS
FMLoss1 = lambdaFM * pix2pixHDFeatureMatchingLoss(realPred1, fakePred1);
FMLoss2 = lambdaFM * pix2pixHDFeatureMatchingLoss(realPred2, fakePred2);

% VGG LOSS 
realRGB = repmat(realImage, [1 1 3 1]);
genRGB  = repmat(genImage,  [1 1 3 1]);
VGGLoss = lambdaVGG * pix2pixHDVGGLoss(realRGB, genRGB, netVGG);

% GENERATOR TOTAL LOSS
lossGTotal = lambdaGen * (GLossScale1 + GLossScale2 + FMLoss1 + FMLoss2 + VGGLoss + full3dLoss);
gradParamsG = dlgradient(lossGTotal, generator.Learnables, RetainData=true);

% DISCRIMINATOR TOTAL LOSS
lossDTotal = lambdaDisc * (DLossScale1 + DLossScale2) * 0.5;
gradParamsDScale1 = dlgradient(lossDTotal, discScale1.Learnables, RetainData=true);
gradParamsDScale2 = dlgradient(lossDTotal, discScale2.Learnables);

% OUTPUT 
lossD     = gather(extractdata(lossDTotal));
lossGGAN  = gather(extractdata(GLossScale1 + GLossScale2));
lossGFM   = gather(extractdata(FMLoss1 + FMLoss2));
lossGVGG  = gather(extractdata(VGGLoss));

scores = {GScore1 + GScore2, DScore1 + DScore2};
end

%% 2D Adversarial Loss
function [DLoss, GLoss, realPredFtrsD, genPredFtrsD, GScore, DScore] = ...
    pix2pixHDAdverserialLoss(realIm, genIm, discriminator)

% Convert grayscale to RGB
realIm = repmat(realIm, [1 1 3 1]);
genIm  = repmat(genIm,  [1 1 3 1]);

featureNames = ["act_top","act_mid_1","act_mid_2","act_tail","conv2d_final"];

% Get the feature maps for the real image from the discriminator
realPredFtrsD = cell(size(featureNames));
[realPredFtrsD{:}] = forward(discriminator,realIm,Outputs=featureNames);

% Get the feature maps for the generated image from the discriminator
genPredFtrsD = cell(size(featureNames));
[genPredFtrsD{:}] = forward(discriminator,genIm,Outputs=featureNames);

realPred = realPredFtrsD{end};
fakePred = genPredFtrsD{end};

GScore = mean(sigmoid(fakePred), "all");
DScore = 0.5*mean(sigmoid(realPred),"all") + ...
    0.5*mean(1 - sigmoid(fakePred),"all");

DLoss = mean((1 - realPred).^2 + fakePred.^2, "all");
GLoss = mean((1 - fakePred).^2, "all");
end

%% 2D Feature Matching Loss
function fmLoss = pix2pixHDFeatureMatchingLoss(realF, fakeF)
fmLoss = 0;
for i = 1:numel(realF)
    fmLoss = fmLoss + mean(abs(realF{i} - fakeF{i}), "all");
end
end

%% 2D VGG Loss
function vggLoss = pix2pixHDVGGLoss(realIm, genIm, netVGG)

featureNames = ["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"];
weights = [1/32 1/16 1/8 1/4 1];

realF = cell(size(featureNames));
[realF{:}] = forward(netVGG, realIm, Outputs=featureNames);

genF = cell(size(featureNames));
[genF{:}]  = forward(netVGG, genIm,  Outputs=featureNames);

vggLoss = 0;
for i = 1:numel(realF)
    vggLoss = vggLoss + weights(i) * mean(abs(realF{i} - genF{i}), "all");
end
end


%% Initilialize Validation figure
function [figureHandle, tileHandle, imageAxes1,imageAxes2,imageAxes3, scoreAxesY, LossAxesY, ...
    lineScoreGenXToY, lineScoreDiscY, lineValLossX2Y] = initializeTrainingPlot()

% Create a wide figure for displaying training progress
figureHandle = figure('Name','Training Progress (SupGAN)');
figureHandle.Position(3) = 1.5 * figureHandle.Position(3); % Make figure wider

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
lineScoreGenXToY = animatedline(scoreAxesY, 'Color', 'r', 'LineWidth', 1.5);
lineScoreDiscY   = animatedline(scoreAxesY, 'Color', 'g', 'LineWidth', 1.5);
lineValLossX2Y   = animatedline(LossAxesY, 'Color', 'b', 'Marker', '*', 'LineWidth', 1.2);
end


%% Generate VALIDATION IMAGES and RESULTS
function [genImage, ssim_val] = displayGeneratedValImages(dsVal, imageAxes1, imageAxes2, imageAxes3, LossAxesY, lineValLossX2Y,...
    genXToY, iteration, Nslices)

% Read mini-batch of data
[imX, imY] = next(dsVal);
seg = imX(:,:,:,2,:); seg = extractdata(gather(seg > 0));
imX = single(imX(:,:,:,1,:)); % convert to single for faster trian - GPU
imY = single(imY); % convert to single for faster trian - GPU

% convert to dlarray
imX = dlarray(imX,'SSSCB');
imY= dlarray(imY,'SSSCB');

% Generate images using the generator
imYGenerated = GenerateFullData(imX, genXToY, Nslices);
% Extract image data
inpImage = extractdata(gather(imX));
realImage = extractdata(gather(imY));
genImage = extractdata(gather(imYGenerated));

b = randi([1, size(imYGenerated,5)], 1); % Random batch index

% Find mid slices based on segmentation mass
sumOverYZ = squeeze(sum(sum(seg, 3), 2));
midX = find(sumOverYZ == max(sumOverYZ), 1, 'first');

sumOverXZ = squeeze(sum(sum(seg, 3), 1));
midY = find(sumOverXZ == max(sumOverXZ), 1, 'first');

sumOverXY = squeeze(sum(sum(seg, 2), 1));
midZ = find(sumOverXY == max(sumOverXY), 1, 'first');

% Axial view
axialInp     = inpImage(:,:,midZ,b);
axialReal      = realImage(:,:,midZ,b);
axialGenerated = genImage(:,:,midZ,b);

% Sagittal view
sagittalInp      = flipud(permute(squeeze(inpImage(midX,:,:,b)), [2 1]));
sagittalReal      = flipud(permute(squeeze(realImage(midX,:,:,b)), [2 1]));
sagittalGenerated = flipud(permute(squeeze(genImage(midX,:,:,b)), [2 1]));

% Coronal view
coronalInp      = flipud(permute(squeeze(inpImage(:,midY,:,b)), [2 1]));
coronalReal      = flipud(permute(squeeze(realImage(:,midY,:,b)), [2 1]));
coronalGenerated = flipud(permute(squeeze(genImage(:,midY,:,b)), [2 1]));

% ---- Plot axial ----
cla(imageAxes1);
imageResults = gather(cat(2, axialInp, axialReal, axialGenerated));
imshow(imageResults, [], 'Parent', imageAxes1);
axis(imageAxes1, 'image'); % keep aspect ratio
title(imageAxes1, "Axial: T1 (Left) Vs Real (Mid) vs Generated (Right)");

% ---- Plot sagittal ----
cla(imageAxes2);
imageResults = gather(cat(2, sagittalInp, sagittalReal, sagittalGenerated));
imshow(imageResults, [], 'Parent', imageAxes2);
axis(imageAxes2, 'image');
title(imageAxes2, "Coronal: T1 (Left) Vs Real (Mid) vs Generated (Right)");

% ---- Plot coronal ----
cla(imageAxes3);
imageResults = gather(cat(2, coronalInp, coronalReal, coronalGenerated));
imshow(imageResults, [], 'Parent', imageAxes3);
axis(imageAxes3, 'image');
title(imageAxes3, "Sagittal: T1 (Left) Vs Real (Mid) vs Generated (Right)");

% Compute loss
ssim_val = multissim3(realImage, genImage);

% Update the validation loss plot
addpoints(lineValLossX2Y, iteration, double(1-ssim_val));
legend(LossAxesY, 'ValLoss');

drawnow;
pause(0.05);
end

%% Create full image from slices
function PredFull = GenerateFullData(Inp, genXToY, Nslices)
% Ensure input is on GPU if needed
if canUseGPU
    Inp = gpuArray(Inp);
end
% Remove singleton dimensions
Inp = squeeze(Inp);  % H x W x D

[H, W, D] = size(Inp);
pad = floor(Nslices/2);

% Pad input depth-wise
padded = padarray(Inp, [0, 0, pad], -1, 'both');  % or -1 padding

% Preallocate input batch
inputBatch = zeros(H, W, Nslices, D, 'like', Inp);

% Vectorized patch extraction
for z = 1:D
    inputBatch(:,:,:,z) = padded(:,:,z : z + Nslices - 1);
end

% Reshape and create dlarray
dlX = dlarray(single(inputBatch), 'SSCB');
% Predict in batch
dlY = predict(genXToY, dlX);  % Expected output: H x W x 1 x D

% Reshape result
dlY = squeeze(dlY);  % H x W x D
PredFull = dlY;
end

%% Crop 3D volume to Tumor ROI
function [realCrop, genCrop] = cropToForeground3D(seg, realVol, genVol)

MIN_HW = 32; MIN_D  = 16;   % minimal depth, adjust as needed
MULT = 8;
if isa(seg,'dlarray'), seg = extractdata(seg); end
if isa(seg,'gpuArray'), seg = gather(seg); end
[Htot, Wtot, Dtot, ~, B] = size(seg);

% Global bounding box across the batch
allX = []; allY = []; allZ = [];
for b = 1:B
    [y, x, z] = ind2sub([Htot, Wtot, Dtot], find(seg(:,:,:,1,b) > 0));
    allX = [allX; x]; allY = [allY; y]; allZ = [allZ; z];
end

if isempty(allX)
    % No foreground found
    realCrop = realVol;
    genCrop  = genVol;
else
    xmin = min(allX); xmax = max(allX);
    ymin = min(allY); ymax = max(allY);
    zmin = min(allZ); zmax = max(allZ);   
    % Enforce MIN dimensions    
    H = ymax - ymin + 1;
    if H < MIN_HW
        grow = MIN_HW - H;
        ymin = max(1, ymin - floor(grow/2));
        ymax = min(Htot, ymax + ceil(grow/2));
    end
    W = xmax - xmin + 1;
    if W < MIN_HW
        grow = MIN_HW - W;
        xmin = max(1, xmin - floor(grow/2));
        xmax = min(Wtot, xmax + ceil(grow/2));
    end

    D = zmax - zmin + 1;
    if D < MIN_D
        grow = MIN_D - D;
        zmin = max(1, zmin - floor(grow/2));
        zmax = min(Dtot, zmax + ceil(grow/2));
    end

    % Adjust to multiples of 8 - Compatible to network input    
    H = ymax - ymin + 1; remH = mod(H, MULT);
    if remH ~= 0
        need = MULT - remH;
        ymin = max(1, ymin - floor(need/2));
        ymax = min(Htot, ymax + ceil(need/2));
    end
    W = xmax - xmin + 1; remW = mod(W, MULT);
    if remW ~= 0
        need = MULT - remW;
        xmin = max(1, xmin - floor(need/2));
        xmax = min(Wtot, xmax + ceil(need/2));
    end
    D = zmax - zmin + 1; remD = mod(D, MULT);
    if remD ~= 0
        need = MULT - remD;
        zmin = max(1, zmin - floor(need/2));
        zmax = min(Dtot, zmax + ceil(need/2));
    end

    % Final crop   
    realCrop = realVol(ymin:ymax, xmin:xmax, zmin:zmax, 1, :);
    genCrop  = genVol (ymin:ymax, xmin:xmax, zmin:zmax, 1, :);
end
end

%% MatRead functions - Input data
function data_new = matReadX(filename)
% Load .mat file
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
% --- Separate MRI image and segmentation ---
% Assuming the 4th dimension: (:,:,:,1) = image, (:,:,:,2) = mask
seg = single(data(:,:,:,2));
data = single(data(:,:,:,1));
% --- Normalize the whole 3D volume to [-1, 1] ---
data = double(data);  % Improve precision before percentile calc
% Compute lower and upper percentiles to limit outliers
inMin = prctile(data(:), 1);
inMax = prctile(data(:), 99);

% Clip values outside [inMin, inMax]
data(data < inMin) = inMin;
data(data > inMax) = inMax;

% Handle flat or uniform data
if inMax <= inMin
    warning('Input data has very low intensity variation. Returning zeros.');
    data_new = zeros(size(data), 'like', data);
else
    % Rescale intensities to [-1, 1]
    data_new = rescale(data, -1, 1, 'InputMin', inMin, 'InputMax', inMax);
end
data_new(:,:,:,2) =  seg;
end
% MatRead functions - Response data
function data_new = matReadY(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data = single(data(:,:,:,2));  % just keep one sequence - 1-t1c, 2-t2 & 3-t2f
% --- Normalize whole 3D volume ---
data = double(data);  % Cast to double for precision if needed
% Compute 1st and 99th percentiles
inMin = prctile(data(:), 1);
inMax = prctile(data(:), 99);
% Clip values outside [inMin, inMax]
data(data < inMin) = inMin;
data(data > inMax) = inMax;
% Ensure inMax > inMin to avoid division by zero or invalid scaling
if inMax == inMin
    warning('Input data has no variation in the selected percentile range.');
else
    data_new = rescale(data, -1, 1, 'InputMax', inMax, 'InputMin', inMin);
end
end

