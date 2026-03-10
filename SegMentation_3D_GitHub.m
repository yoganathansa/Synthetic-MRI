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
D = 'Y:\Syn_MRI\BRaTS 2021\3D\Train\Seg\Input';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
imdsInp = @(x) matReadX(x);
imdsInp = imageDatastore(F,'FileExtensions','.mat','ReadFcn',imdsInp);

D = 'Y:\Syn_MRI\BRaTS 2021\3D\Train\Seg\Response';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
imdsRef = @(x) matReadY(x);
imdsRef = imageDatastore(F,'FileExtensions','.mat','ReadFcn',imdsRef);

% Validation data
D = 'Y:\Syn_MRI\BRaTS 2021\3D\Valid\Seg\Input';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
Funct = @(x) matReadX(x);
imdsValInp = imageDatastore(F,'FileExtensions','.mat','ReadFcn',Funct);

D = 'Y:\Syn_MRI\BRaTS 2021\3D\Valid\Seg\Response';
S = dir(fullfile(D,'*.mat'));
N = natsortfiles({S.name});
F = cellfun(@(n)fullfile(D,n),N,'uni',0);
Funct = @(x) matReadY(x);
imdsValRef = imageDatastore(F,'FileExtensions','.mat','ReadFcn',Funct);

% Extract random patches
miniBatchSize = 2;
% Combine input and response datastores
dsTrain = combine(imdsInp, imdsRef);
dsVal = combine(imdsValInp, imdsValRef);

% Create minibatchqueue from the combined datastore
mbqTrain = minibatchqueue(dsTrain, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@(x,y) deal(concatenateMiniBatchX(x), concatenateMiniBatchY(y)), ...
    PartialMiniBatch="discard", ...
    MiniBatchFormat=["SSSCB", "SSSCB"]);

% Create minibatchqueue from the combined datastore
mbqVal = minibatchqueue(dsVal, ...
    MiniBatchSize=1, ...
    MiniBatchFcn=@(x,y) deal(concatenateMiniBatchX(x), concatenateMiniBatchY(y)), ...
    PartialMiniBatch="discard", ...
    MiniBatchFormat=["SSSCB", "SSSCB"]);
%% Network Architecture
inputSize = [208 160 144 4];
classWeights = [0.008 2.72 0.68 1.86];
genX2Y = createEncoderDecoder3D(inputSize,16,3,5,1,4);
% Replace layers for segmentation task
inpLayer = image3dInputLayer(inputSize,"Name",'Inp','Normalization','none');
genX2Y = replaceLayer(genX2Y, genX2Y.Layers(1,1).Name, inpLayer);
SMLayer = softmaxLayer("Name",'SoftMax_Output');
genX2Y = replaceLayer(genX2Y, 'output_Block_Tanh', SMLayer);
genX2Y = dlnetwork(genX2Y);

%% Training
numEpochs = 15;
LearnRate = 0.00005;
trailAvgGen = []; trailAvgSqGen = [];
gradDecayFac = 0.5; sqGradDecayFac = 0.999;

validFreq = 500; epoch = 0; iteration = 0; start = tic;

% Monitor training and validation progress
monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss");

% specify log y-axis scale - to switch scale during training, 

monitor.Info=["LearningRate","Epoch","Iteration","ExecutionEnvironment"];
monitor.XLabel = "Iteration";
monitor.Status = "Configuring";
monitor.Progress = 0;

% Initialize plots for training progress
[figureHandle, tileHandle, imageAxes1, imageAxes2, imageAxes3,...
    lossSeg1, lossSeg2, lossSeg3, lineSeg1, lineSeg2, lineSeg3] = initializeTrainingPlot();
% Make sure figure is visible (especially in Live Script)
set(figureHandle, 'Visible', 'on');

numObservationsTrain = numel(dsTrain.UnderlyingDatastores{1,1}.Files);
numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

dataDir = fullfile('Y:\Syn_MRI\');
% Create a directory to store checkpoints
checkpointDir = fullfile(dataDir,"Models");
if ~exist(checkpointDir,"dir")
    mkdir(checkpointDir);
end

executionEnvironment = "gpu";
monitor.Status = "Running";

% Training loop
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    % Reset and shuffle the data
    reset(mbqTrain);
    shuffle(mbqTrain);
    % Loop over mini-batches
    while hasdata(mbqTrain) && ~monitor.Stop
        iteration = iteration + 1;
        % Read mini-batch of data
        [Inp, Output] = next(mbqTrain);
        Inp = single(Inp);
        Output = single(Output);

        % Calculate the loss and gradients
        [gradParamsG, Loss] = ...
            dlfeval(@modelGradients, Inp, Output, genX2Y, classWeights);

        % Update parameters of generator
        [genX2Y, trailAvgGen, trailAvgSqGen] = adamupdate( ...
            genX2Y, gradParamsG, trailAvgGen, trailAvgSqGen, iteration, ...
            LearnRate, gradDecayFac, sqGradDecayFac);

        % Validation Loss
        if mod(iteration,validFreq) == 0 || iteration == 1
            % Reset Validation data
            if hasdata(mbqVal) == 0
                reset(mbqVal); shuffle(mbqVal);
            end
            [ValLossAll, DiceLosSeg2, DiceLosSeg3, DiceLosSeg4] ...
                = Validation_Seg(mbqVal, genX2Y, imageAxes1, imageAxes2, imageAxes3);

            % Update the validation loss line plot
            addpoints(lineSeg1, iteration, double(DiceLosSeg2))
            legend(lossSeg1, 'NT');

            addpoints(lineSeg2, iteration, double(DiceLosSeg3))
            legend(lossSeg2, 'ED');

            addpoints(lineSeg3, iteration, double(DiceLosSeg4))
            legend(lossSeg3, 'ET');

            drawnow expose;     % Force GUI refresh
            pause(0.05);        % Ensure Live Script updates figure
        end

        % --- Update the monitor ---
        recordMetrics(monitor, iteration, ...
            TrainingLoss=Loss);

        monitor.Info=["LearningRate","Epoch","Iteration","ExecutionEnvironment"];

        updateInfo(monitor,LearningRate=LearnRate, ...
            Epoch=epoch, ...
            Iteration=iteration, ExecutionEnvironment ="GPU");

            monitor.Progress = 100 * iteration/numIterations;
    end
end

% Save the final model
modeEateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
save(checkpointDir+filesep+"segMRI-"+modeEateTime+".mat",'genX2Y');

%% MatRead - Input Function
function data = matReadX(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data = single(data);  %keep only one image - excluding mask-seg
% --- Normalize whole 3D volume ---
for i=1:size(data,4)
    d = data(:,:,:,i);
    % Compute 1st and 99th percentiles
    inMin = prctile(d(:), 1);
    inMax = prctile(d(:), 99);
    % Ensure inMax > inMin to avoid division by zero or invalid scaling
    if inMax == inMin
        warning('Input data has no variation in the selected percentile range.');
    else
        data(:,:,:,i) = rescale(d, -1, 1, 'InputMax', inMax, 'InputMin', inMin);
    end
end
end
% MatRead - Response Function
function data_new = matReadY(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data(data>2)=3;
data = uint8(data+1);

% convert to one-hot encoded
inputSize = size(data);
H = inputSize(1);
W = inputSize(2);
D = inputSize(3);
C = 4; % Number of classes

% Preallocate one-hot encoded array
data_new = zeros(H, W, D, C, 'like', data);

% Loop over classes and create binary mask
for classIdx = 1:C
    data_new(:,:,:,classIdx) = data == classIdx;
end
end


%% Model Gradients & Loss

function [gradParamsG, Loss] = ...
    modelGradients(input, realImage, generator, classWeights)

% Compute the image generated by the generator given the input semantic map.
predVol = forward(generator,input);

alpha = 0.3; beta = 0.7; 
lambdaDice = 7; lambdaTversky = 2; lambdaCE = 1;

Loss = combinedSegLoss(predVol, realImage, classWeights, alpha, beta, ...
                       lambdaDice, lambdaTversky, lambdaCE);

% Compute gradients for the generator
gradParamsG = dlgradient(Loss,generator.Learnables,RetainData=true);
end

function loss = combinedSegLoss(Ypred, Ytrue, classWeights, alpha, beta, lambdaDice, lambdaTversky, lambdaCE)
% Combined Weighted Loss for Multi-class Segmentation
% Includes Weighted Dice + Tversky + Cross-Entropy losses.
%
% Inputs:
%   Ypred        : HxWxDxCxN (softmax probabilities)
%   Ytrue        : HxWxDxCxN (one-hot ground truth)
%   classWeights : [1xC] class weights
%   alpha, beta  : Tversky parameters (e.g., alpha=0.5, beta=0.5)
%   lambdaDice, lambdaTversky, lambdaCE : weighting coefficients for loss components
%
% Output:
%   loss : scalar combined loss

smooth = 1e-6; numClasses = size(Ypred, 4); batchSize  = size(Ypred, 5);

diceLoss   = zeros(batchSize, numClasses, 'like', Ypred);
tverskyLoss = zeros(batchSize, numClasses, 'like', Ypred);
ceLoss      = zeros(batchSize, numClasses, 'like', Ypred);

for n = 1:batchSize
    for c = 1:numClasses
        Yp = Ypred(:,:,:,c,n);
        Yt = Ytrue(:,:,:,c,n);

        % --- Dice ---
        intersection = sum(Yp .* Yt, 'all');
        denom = sum(Yp, 'all') + sum(Yt, 'all');
        diceScore = (2 * intersection + smooth) / (denom + smooth);
        diceLoss(n,c) = 1 - diceScore;

        % --- Tversky ---
        TP = intersection;
        FP = sum(Yp .* (1 - Yt), 'all');
        FN = sum((1 - Yp) .* Yt, 'all');
        tverskyIndex = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth);
        tverskyLoss(n,c) = 1 - tverskyIndex;

        % --- Cross Entropy ---
        ce = -Yt .* log(Yp + smooth);  % avoid log(0)
        ceLoss(n,c) = mean(ce, 'all');
    end
end

% Average over batch
meanDicePerClass    = mean(diceLoss, 1);
meanTverskyPerClass = mean(tverskyLoss, 1);
meanCEPerClass      = mean(ceLoss, 1);

% Apply class weights
weightedDice    = sum(classWeights .* meanDicePerClass)    / sum(classWeights);
weightedTversky = sum(classWeights .* meanTverskyPerClass) / sum(classWeights);
weightedCE      = sum(classWeights .* meanCEPerClass)      / sum(classWeights);

% Combine total loss
loss = lambdaDice * weightedDice + lambdaTversky * weightedTversky + lambdaCE * weightedCE;
end
%% Minibatch Processor
function X = concatenateMiniBatchX(im1)
% Concatenate a mini-batch of data along the batch dimension.
X = cat(5,im1{:});
end
function Y = concatenateMiniBatchY(im1)
% Concatenate a mini-batch of data along the batch dimension.
Y = cat(5,im1{:});
end
%% Generate Validation Images and Results
function  [ValLossAll, DiceLosSeg2, DiceLosSeg3, DiceLosSeg4] ...
    = Validation_Seg(mbqVal, genX2Y, imageAxes1, imageAxes2, imageAxes3)

% Read mini-batch of data
[Inp, output] = next(mbqVal);

smooth = 1e-6;
% Initialize arrays to store per-class Dice losses
DiceLosSeg2 = zeros(1, size(Inp,5));
DiceLosSeg3 = zeros(1, size(Inp,5));
DiceLosSeg4 = zeros(1, size(Inp,5));

% Initialize array to store combined mean loss per volume
ValLossAll = zeros(1, size(Inp,5));

for ii = 1:size(Inp,5)
    % Load input and ground truth volume
    inpVol = Inp(:,:,:,:,ii);
    trueVol = output(:,:,:,:,ii);

    % Get prediction from model
    predVol = forward(genX2Y, inpVol);

    perClassLoss = zeros(1, 4);  % For classes 1–4

    for classIdx = 1:4
        predClass = predVol(:,:,:,classIdx);
        trueClass = trueVol(:,:,:,classIdx);

        intersection = sum(predClass .* trueClass, 'all');
        denom = sum(predClass, 'all') + sum(trueClass, 'all');
        diceScore = (2 * intersection + smooth) / (denom + smooth);
        diceLoss = 1 - diceScore;

        perClassLoss(classIdx) = diceLoss;

        % Store class-specific losses
        switch classIdx
            case 2
                DiceLosSeg2(ii) = diceLoss;
            case 3
                DiceLosSeg3(ii) = diceLoss;
            case 4
                DiceLosSeg4(ii) = diceLoss;
        end
    end

    % Combined mean loss across all 4 classes (including background)
    ValLossAll(ii) = mean(perClassLoss);
end

% Final average losses
DiceLosSeg2 = mean(DiceLosSeg2);
DiceLosSeg3 = mean(DiceLosSeg3);
DiceLosSeg4 = mean(DiceLosSeg4);
ValLossAll = mean(ValLossAll);

%  Visualization 

% Extract image data from last volume for visualization
inpImages       = extractdata(gather(inpVol));         % [208 x 160 x 144 x 4]
realImages      = extractdata(gather(trueVol));        % one-hot
generatedImage  = extractdata(gather(predVol));        % one-hot

% Select modality 2 (e.g., T1) from input
inpImages = inpImages(:,:,:,2);  % [208 x 160 x 144]

% Convert one-hot to label map
[~, realLabels] = max(realImages, [], 4);
[~, predLabels] = max(generatedImage, [], 4);

% Create binary mask for all foreground classes (1..4)
mask = realLabels > 1;

% Find mid slices based on segmentation mass
sumOverYZ = squeeze(sum(sum(mask, 3), 2));
midX = find(sumOverYZ == max(sumOverYZ), 1, 'first');

sumOverXZ = squeeze(sum(sum(mask, 3), 1));
midY = find(sumOverXZ == max(sumOverXZ), 1, 'first');

sumOverXY = squeeze(sum(sum(mask, 2), 1));
midZ = find(sumOverXY == max(sumOverXY), 1, 'first');


% Create RGB overlays
% Define colormap for 4 classes
cmap = lines(4);  % or use your custom 4-class map

% Normalize input volumes for display
axialInp     = rescale(inpImages(:,:,midZ));
sagittalInp  = rescale(flipud(permute(inpImages(midX,:,:), [3 2 1])));
coronalInp   = rescale(flipud(permute(inpImages(:,midY,:), [3 1 2])));

% Convert label maps to 2D slices
axialReal      = realImages(:,:,midZ);
axialGenerated = predLabels(:,:,midZ);

sagittalReal      = flipud(permute(realLabels(midX,:,:), [3 2 1]));
sagittalGenerated = flipud(permute(predLabels(midX,:,:), [3 2 1]));

coronalReal      = flipud(permute(realLabels(:,midY,:), [3 1 2]));
coronalGenerated = flipud(permute(predLabels(:,midY,:), [3 1 2]));

% Plot overlays 

% Axial view
cla(imageAxes1);
overlayGT  = labeloverlay(axialInp, axialReal, 'Colormap', cmap, 'Transparency', 0.5);
overlayPred = labeloverlay(axialInp, axialGenerated, 'Colormap', cmap, 'Transparency', 0.5);
imshow([overlayGT, overlayPred], 'Parent', imageAxes1);
axis(imageAxes1, 'image');
title(imageAxes1, "Axial: Ground Truth (Left) | Prediction (Right)");

% Sagittal view
cla(imageAxes2);
overlayGT  = labeloverlay(sagittalInp, sagittalReal, 'Colormap', cmap, 'Transparency', 0.5);
overlayPred = labeloverlay(sagittalInp, sagittalGenerated, 'Colormap', cmap, 'Transparency', 0.5);
imshow([overlayGT, overlayPred], 'Parent', imageAxes2);
axis(imageAxes2, 'image');
title(imageAxes2, "Sagittal: Ground Truth (Left) | Prediction (Right)");

% Coronal view
cla(imageAxes3);
overlayGT  = labeloverlay(coronalInp, coronalReal, 'Colormap', cmap, 'Transparency', 0.5);
overlayPred = labeloverlay(coronalInp, coronalGenerated, 'Colormap', cmap, 'Transparency', 0.5);
imshow([overlayGT, overlayPred], 'Parent', imageAxes3);
axis(imageAxes3, 'image');
title(imageAxes3, "Coronal: Ground Truth (Left) | Prediction (Right)");

end

%% Initilialize figure
function [figureHandle, tileHandle, imageAxes1,imageAxes2, imageAxes3,...
    lossSeg1, lossSeg2, lossSeg3, lineSeg1, lineSeg2, lineSeg3] = initializeTrainingPlot()
% Initialize figure layout for SupGAN training

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

% Tile 4: Seg1 Loss
lossSeg1 = nexttile(tileHandle, 4);
xlabel(lossSeg1, "Iteration");
ylabel(lossSeg1, "Loss");
grid(lossSeg1, "on");

% Tile 5: Seg2 Loss
lossSeg2 = nexttile(tileHandle, 5);
xlabel(lossSeg2, "Iteration");
ylabel(lossSeg2, "Score");
grid(lossSeg2, "on");

% Tile 6: Validation loss
lossSeg3 = nexttile(tileHandle, 6);
xlabel(lossSeg3, "Iteration");
ylabel(lossSeg3, "Loss");
grid(lossSeg3, "on");

% Initialize animated lines
lineSeg2 = animatedline(lossSeg2, 'Color', 'r', 'LineWidth', 1.5);
lineSeg3  = animatedline(lossSeg3, 'Color', 'g', 'LineWidth', 1.5);
lineSeg1  = animatedline(lossSeg1, 'Color', 'b', 'Marker', '*', 'LineWidth', 1.5);
end

%% Generator Architecture
function net = createEncoderDecoder3D(imageSize,initialNumFilters,numDownSamp,numResBlock,add_atten,numOPChannel)
% initialNumFilters = 64; numDownSamp = 4; numResBlock = 9;
net = layerGraph;
% input block
inpBlock= [
    image3dInputLayer(imageSize,"Name","inp_Block_Inp","Normalization","zerocenter")
    convolution3dLayer(3,initialNumFilters,"Name","inp_Block_Conv1","Padding",'same',...
    "WeightsInitializer","narrow-normal")
    instanceNormalizationLayer("Name","inp_Block_InstNorm1")
    reluLayer("Name","inp_Block_Relu1")
    convolution3dLayer(3,initialNumFilters,"Name","inp_Block_Conv2","Padding",'same',...
    "WeightsInitializer","narrow-normal")
    instanceNormalizationLayer("Name","inp_Block_InstNorm2")
    reluLayer("Name","inp_Block_Relu2")
    ];
net = addLayers(net,inpBlock);

% encoder block
for i = 1:numDownSamp; filt(i) = initialNumFilters * 2^i; end

for i = 1:numDownSamp
    layers = [
        convolution3dLayer(3, filt(i),"Padding",'same', ...
        "Name", ['encoder_Block' num2str(i) '_Conv1'], 'Stride',2, "WeightsInitializer", "narrow-normal")
        instanceNormalizationLayer("Name", ['encoder_Block' num2str(i) '_InstNorm1'])
        reluLayer("Name", ['encoder_Block' num2str(i) '_Relu1'])
        convolution3dLayer(3, filt(i),  "Padding", 'same', ...
        "Name", ['encoder_Block' num2str(i) '_Conv2'],"WeightsInitializer", "narrow-normal")
        instanceNormalizationLayer("Name", ['encoder_Block' num2str(i) '_InstNorm2'])
        reluLayer("Name", ['encoder_Block' num2str(i) '_Relu2'])
        ];
    net = addLayers(net, layers);
    if i>1 % Connecting encoders
        net = connectLayers(net, ['encoder_Block' num2str(i-1) '_Relu2'],['encoder_Block' num2str(i) '_Conv1'] );
    end
end

%connecting inputblock and encoder
net = connectLayers(net,"inp_Block_Relu2",'encoder_Block1_Conv1');

% Residual block
res_filt = filt(end);

for i = 1:numResBlock
    layers = [convolution3dLayer(3,res_filt,"Padding",'same',...
        "Name", ['resB_Block' num2str(i) '_Conv1'],"WeightsInitializer","narrow-normal")
        instanceNormalizationLayer("Name", ['resB_Block' num2str(i) '_InstNorm1'])
        reluLayer("Name", ['resB_Block' num2str(i) '_Relu'])
        convolution3dLayer(3,res_filt,"Padding",'same',...
        "Name", ['resB_Block' num2str(i) '_Conv2'], "WeightsInitializer","narrow-normal")
        instanceNormalizationLayer("Name", ['resB_Block' num2str(i) '_InstNorm2'])
        additionLayer(2,"Name", ['resB_Block' num2str(i) '_Add'])];

    net = addLayers(net, layers);
    % connecting ResBlocks
    if i>1
        net = connectLayers(net, ['resB_Block' num2str(i-1) '_Add'],['resB_Block' num2str(i) '_Conv1'] );
    end
end
% connecting encoder and Bridge
net = connectLayers(net,['encoder_Block' num2str(numDownSamp) '_Relu2'],'resB_Block1_Conv1');
net = connectLayers(net,['encoder_Block' num2str(numDownSamp) '_Relu2'],'resB_Block1_Add/in2');

% connections within ResBlocks
for i=1:numResBlock-1
    net = connectLayers(net,['resB_Block' num2str(i) '_Add'],['resB_Block' num2str(i+1) '_Add/in2']);
end
% decoder block
for i = numDownSamp:-1:1
    filt = initialNumFilters * 2^(i-1);
    layers = [transposedConv3dLayer(2,filt,"Stride",2,...
        "Name", ['decoder_Block' num2str(i) '_TranspConv'],"WeightsInitializer","narrow-normal")
        instanceNormalizationLayer("Name", ['decoder_Block' num2str(i) '_InstNorm1'])
        reluLayer("Name", ['decoder_Block' num2str(i) '_Relu1'])

        convolution3dLayer(3,filt, "Padding", 'same',...
        "Name",['decoder_Block' num2str(i) '_Conv'],"WeightsInitializer", "narrow-normal")
        instanceNormalizationLayer("Name", ['decoder_Block' num2str(i) '_InstNorm2'])
        reluLayer("Name", ['decoder_Block' num2str(i) '_Relu2'])

        concatenationLayer(4, 2, 'Name', ['decoder_Block' num2str(i) '_Concat']);];

    net = addLayers(net, layers);

    if i<numDownSamp % connecting decoder blocks
        net = connectLayers(net, ['decoder_Block' num2str(i+1) '_Concat'],['decoder_Block' num2str(i) '_TranspConv'] );
    end
end

% final output block
layers = [convolution3dLayer(1,numOPChannel,"Name","output_Block_Conv","WeightsInitializer","narrow-normal")
    tanhLayer("Name","output_Block_Tanh")];
net = addLayers(net, layers);

% connections decoder and output block
net = connectLayers(net,'decoder_Block1_Concat','output_Block_Conv');

% connections ResBlock and decoder
net = connectLayers(net,['resB_Block' num2str(numResBlock) '_Add'],['decoder_Block' num2str(numDownSamp) '_TranspConv']);

% Add attention
if add_atten==1
    for i=1:numDownSamp
        filt = initialNumFilters* 2^(i-1);
        x =      [convolution3dLayer(1, filt, 'Stride', 1, "Padding", "same", ...
            "Name", ['atten_Block' num2str(i) '_Encoder-Conv'], "WeightsInitializer", "narrow-normal")
            crop3dLayer("Name",['atten_Block' num2str(i) '_Encoder-Crop'])
            additionLayer(2,"Name", ['atten_Block' num2str(i) '_Add'])];
        net = addLayers(net,x);

        g = [convolution3dLayer(1, filt, 'Stride', 1, "Padding", "same", ...
            "Name", ['atten_Block' num2str(i) '_Decoder-Conv'], "WeightsInitializer", "narrow-normal")];
        net = addLayers(net,g);
        % add X (Encoder) and G (Decoder)
        net = connectLayers(net, ['atten_Block' num2str(i) '_Decoder-Conv'],['atten_Block' num2str(i) '_Add/in2']);
        % Crop Encoder output
        net = connectLayers(net, ['atten_Block' num2str(i) '_Decoder-Conv'],['atten_Block' num2str(i) '_Encoder-Crop/ref']);
        % Path after Merging X (Encoder) and G (Decoder)
        layers = [reluLayer("Name",['atten_Block' num2str(i) '_Merged-Relu'])
            convolution3dLayer(1, 1, 'Stride', 1, "Padding", "same", ...
            "Name", ['atten_Block' num2str(i) '_Merged-Conv'], "WeightsInitializer", "narrow-normal")
            sigmoidLayer("Name",['atten_Block' num2str(i) '_Merged-Sigmoid'])            
            multiplicationLayer(2,"Name",['atten_Block' num2str(i) '_Merged-Multi'])];

        net = addLayers(net,layers);
        % connect Merged Path
        net = connectLayers(net, ['atten_Block' num2str(i) '_Add/out'],['atten_Block' num2str(i) '_Merged-Relu']);
        % Crop layer for skip connection
        layers = crop3dLayer("Name",['atten_Block' num2str(i) '_Skip-Crop']);
        net = addLayers(net,layers);
        % connect Skip crop to Ref
        net = connectLayers(net, ['atten_Block' num2str(i) '_Merged-Sigmoid'],['atten_Block' num2str(i) '_Skip-Crop/ref']);% Skip crop to atten multi  
    end

    % Connect Encoder to X - attention layers
    for i=1:numDownSamp
        if i==1
            net = connectLayers(net,'inp_Block_Relu2',['atten_Block' num2str(i) '_Encoder-Conv']); % connect Encoder - Attention
            net = connectLayers(net,'inp_Block_Relu2',['atten_Block' num2str(i) '_Skip-Crop/in']);  % Encoder Skip connection  to crop
             net = connectLayers(net,['atten_Block' num2str(i) '_Skip-Crop'],['atten_Block' num2str(i) '_Merged-Multi/in2']);  % Skip crop to atten multi          
        else
            net = connectLayers(net, ['encoder_Block' num2str(i-1) '_Relu2'],['atten_Block' num2str(i) '_Encoder-Conv']); % connect Encoder - Attention 
            net = connectLayers(net, ['encoder_Block' num2str(i-1) '_Relu2'],['atten_Block' num2str(i) '_Skip-Crop/in']);% Encoder Skip connection  to crop
            net = connectLayers(net, ['atten_Block' num2str(i) '_Skip-Crop'],['atten_Block' num2str(i) '_Merged-Multi/in2']);% Skip crop to atten multi    
        end
    end

    % Connect attention output to depthconcatenation of decoder
    for i=1:numDownSamp
        net = connectLayers(net,['atten_Block' num2str(i) '_Merged-Multi'],['decoder_Block' num2str(i) '_Concat/in2']);
    end

    % connect decoder to G-attention layers % connect attention output to decoder depthconcatanation
    for i=numDownSamp:-1:1
        net = connectLayers(net,['decoder_Block' num2str(i) '_Relu2'],['atten_Block' num2str(i) '_Decoder-Conv']);
    end
end
% Initialize network
% net = dlnetwork(net);
end