clear; clc; close all;

%% ── RNG ──────────────────────────────────────────────────────────────────
rngSeed = 42;
rng(rngSeed, "twister");

%% ── 路徑設定 ─────────────────────────────────────────────────────────────
thisFile = mfilename("fullpath");
projectRoot = fileparts(thisFile);
if strlength(projectRoot) == 0
    projectRoot = pwd;
end

labelFile = fullfile(projectRoot, "labels", "YOUR_LABEL_FILE.xlsx");
outDir    = fullfile(projectRoot, "results");

if ~isfolder(outDir)
    mkdir(outDir);
end

%% ── 使用者參數 ───────────────────────────────────────────────────────────
defaultFS = 5000;

% 前處理
useBandpass    = true;   bpHz       = [50 500];
useNotch       = true;   notchesHz  = [60 120 180];
doDetrend      = true;
doMedianCenter = true;
doClip99       = false;
doRectify      = false;
doNormPerWin   = false;

% 活動門控（Activity gate）
useActivityGate    = true;
envMethod          = "arv";   
envWinMs           = 20;
gateK              = 3.0;
minActiveFracInWin = 0.10;
gateFallbackKeepAll = true;

% 切窗
winSec       = 0.3;
hopRatio     = 0.75;
keepTopRMSFrac = 1.00;   

% 每隻老鼠上限
capPerRatTrain = 200;
capPerRatTest  = 200;

% 1D CNN 
maxEpochs1D = 15;  miniBatch1D = 32;

% 2D CNN  STFT
imgH = 64;  imgW = 64;  stftBandHz = [20 500];
maxEpochs2D = 15;  miniBatch2D = 32;

% Rat-level validation 分割
nValRatsPerClass = 2;

% 示範圖片輸出
saveExamplePlots = true;
exampleOutDir    = fullfile(outDir, "example_outputs");

%% ── 輸出檔名 ─────────────────────────────────────────────────────────────
timeTag = string(datetime("now","Format","yyyyMMdd_HHmmss"));
winTag  = sprintf("win%.3fs_hop%.2f", winSec, hopRatio);

windowsMatPath = fullfile(outDir, "eus_windows_"  + winTag + "_" + timeTag + ".mat");
resMatPath     = fullfile(outDir, "eus_RESULTS_"  + winTag + "_" + timeTag + ".mat");
fusePlotPath   = fullfile(outDir, "PerRat_pFuse_" + winTag + "_" + timeTag + ".png");

fprintf("=== EUS-EMG Pipeline | %s | fs=%d ===\n", winTag, defaultFS);

%% ── 0) 讀取標記 ──────────────────────────────────────────────────────────
if ~isfile(labelFile)
    error("Label file not found: %s", labelFile);
end

T = [readLabelSheetForce(labelFile,"Sham"); readLabelSheetForce(labelFile,"MCAO")];
if isempty(T), error("Label table is empty: %s", labelFile); end

T.fs(:)    = defaultFS;
T.rat_id   = extractRatIdFromPath(T.file_path);
T = T(T.rat_id ~= "<UNK>", :);
if height(T) == 0, error("All rat IDs failed to parse (expected rXXXX in filename)."); end

fprintf("Labels loaded: %d bursts\n", height(T));

%% ── 1) 切窗 ──────────────────────────────────────────────────────────────
winN = round(winSec * defaultFS);
hopN = round(winSec * hopRatio * defaultFS);

Xcell      = {};
Ystr       = strings(0,1);
rat_id_all = strings(0,1);
window_uid = strings(0,1);
winRMS     = zeros(0,1);
winEnvMean = zeros(0,1);
winActFrac = zeros(0,1);
segInfo    = table();

nWinTried = 0;  nWinKept = 0;  nBurstUsed = 0;

for i = 1:height(T)
    fp  = T.file_path(i);
    grp = T.group(i);
    rid = T.rat_id(i);
    t0  = T.t0(i);
    t1  = T.t1(i);

    if ~isfile(fp), warning("Missing: %s — skipped.", fp); continue; end

    try
        [emg, fs] = read_emg_ch2(fp, defaultFS);
    catch ME
        warning("Read failed: %s (%s) — skipped.", fp, ME.message); continue;
    end

    i0 = max(1, floor(t0*fs)+1);
    i1 = min(numel(emg), floor(t1*fs));
    if i1 <= i0, continue; end

    seg = double(emg(i0:i1)) - mean(emg(i0:i1),"omitnan");

    % 陷波濾波
    if useNotch
        for hz = notchesHz
            w = [hz-1 hz+1]/(fs/2);
            if any(w<=0)||any(w>=1), continue; end
            [bn,an] = butter(2,w,"stop");
            seg = filtfilt(bn,an,seg);
        end
    end

    % 帶通濾波
    if useBandpass
        [b,a] = butter(4, bpHz/(fs/2), "bandpass");
        seg = filtfilt(b,a,seg);
    end

    if doDetrend,      seg = detrend(seg);      end
    if doMedianCenter, seg = seg - median(seg); end
    if doClip99
        pv = prctile(abs(seg),99);
        if pv>0, seg = max(min(seg,3*pv),-3*pv); end
    end
    if doRectify, seg = abs(seg); end

    % 活動門控
    env = []; actMask = []; thr = NaN;
    if useActivityGate
        [env, actMask, thr] = buildActivityMask(seg, fs, envMethod, envWinMs, gateK);
        if gateFallbackKeepAll && ~any(actMask)
            actMask = true(size(seg));
        end
    end

    nSamp = numel(seg);
    idxList = 1;
    if nSamp >= winN, idxList = 1:hopN:(nSamp-winN+1); end

    nBurstUsed = nBurstUsed + 1;

    for k = 1:numel(idxList)
        stRel = idxList(k);
        nWinTried = nWinTried + 1;

        if nSamp < winN
            segWin0  = [seg(:); zeros(winN-nSamp,1)];
            absStart = i0;  absEnd = i0+winN-1;
            actFrac = useActivityGate * mean(actMask);
            envMean = useActivityGate * mean(env);
        else
            segWin0  = seg(stRel:stRel+winN-1);
            absStart = i0+stRel-1;  absEnd = absStart+winN-1;
            if useActivityGate
                msk     = actMask(stRel:stRel+winN-1);
                actFrac = mean(msk);
                envMean = mean(env(stRel:stRel+winN-1));
            else
                actFrac = NaN;  envMean = NaN;
            end
        end

        if useActivityGate && (actFrac < minActiveFracInWin), continue; end

        segWin = segWin0;
        if doNormPerWin
            segWin = segWin - mean(segWin);
            sd = std(segWin);
            if sd > 1e-12, segWin = segWin./sd; end
        end

        Xcell{end+1,1}      = segWin(:)'; %#ok<AGROW>
        Ystr(end+1,1)       = string(grp);
        rat_id_all(end+1,1) = string(rid);
        uid = rid+"|"+string(i)+"|"+string(k)+"|"+string(absStart)+"-"+string(absEnd);
        window_uid(end+1,1) = uid;
        winRMS(end+1,1)     = sqrt(mean(segWin0.^2));
        winEnvMean(end+1,1) = envMean;
        winActFrac(end+1,1) = actFrac;

        segInfo = [segInfo; table( ...
            string(fp),string(grp),string(rid), ...
            i,k,stRel,absStart,absEnd,t0,t1,uid,envMean,actFrac,thr, ...
            'VariableNames', ...
            ["file_path","group","rat_id","burstRow","winInBurst","stRel", ...
             "absStart","absEnd","t0","t1","window_uid","envMean","actFrac","thr"])]; %#ok<AGROW>

        nWinKept = nWinKept + 1;
    end
end

if isempty(Xcell), error("No windows generated. Check labels / gate settings."); end

Y  = categorical(Ystr, ["Sham","MCAO"]);
fs = defaultFS;
uRat = unique(rat_id_all,"stable");
nRat = numel(uRat);

fprintf("Windows kept: %d / %d (%.1f%%) | bursts=%d | rats=%d\n", ...
    nWinKept, nWinTried, 100*nWinKept/max(1,nWinTried), nBurstUsed, nRat);

save(windowsMatPath, ...
    "Xcell","Y","rat_id_all","window_uid","winRMS","winEnvMean","winActFrac","segInfo", ...
    "fs","winSec","hopRatio","winN","hopN", ...
    "useBandpass","bpHz","useNotch","notchesHz","doDetrend","doMedianCenter", ...
    "doClip99","doRectify","doNormPerWin", ...
    "useActivityGate","envMethod","envWinMs","gateK","minActiveFracInWin","gateFallbackKeepAll", ...
    "rngSeed","keepTopRMSFrac","-v7.3");
fprintf("Saved windows: %s\n", windowsMatPath);

%% ── 2) 建立 STFT 影像 ────────────────────────────────────────────────────
fprintf("Building STFT images...\n");

Ximgs     = cell(numel(Xcell),1);
winSTFT   = hann(256,"periodic");
overlapST = 128;
fftLen    = 512;

for n = 1:numel(Xcell)
    [S, freq] = stft(double(Xcell{n}(:)), fs, ...
        "Window",winSTFT,"OverlapLength",overlapST,"FFTLength",fftLen);
    P     = abs(S).^2;
    Pband = log1p(P(freq>=stftBandHz(1)&freq<=stftBandHz(2), :));
    Ximgs{n} = reshape(single(imresize(Pband,[imgH imgW])), [imgH imgW 1]);
end

fprintf("STFT images built: %d\n", numel(Ximgs));

if saveExamplePlots
    saveExampleFigures(Xcell, Ximgs, Y, winActFrac, fs, stftBandHz, exampleOutDir);
end

%% ── 3) LOSO 折疊劃分 ─────────────────────────────────────────────────────
fprintf("LOSO folds: %d rats\n", nRat);

idxTrFold = cell(nRat,1);
idxTeFold = cell(nRat,1);

for fold = 1:nRat
    isTest  = (rat_id_all == uRat(fold));
    isTrain = ~isTest;
    isTrain = applyCapPerRatByScore(isTrain, rat_id_all, capPerRatTrain, winActFrac);
    isTest  = applyCapPerRatByScore(isTest,  rat_id_all, capPerRatTest,  winActFrac);
    if keepTopRMSFrac < 1.0
        isTrain = applyTopFracRMS(isTrain, rat_id_all, winRMS, keepTopRMSFrac);
        isTest  = applyTopFracRMS(isTest,  rat_id_all, winRMS, keepTopRMSFrac);
    end
    idxTrFold{fold} = find(isTrain);
    idxTeFold{fold} = find(isTest);
end

%% ── 4) 逐折訓練 & 測試 ───────────────────────────────────────────────────
classes = categories(Y);

foldScore1D = cell(nRat,1);  foldYte1D = cell(nRat,1);
foldScore2D = cell(nRat,1);  foldYte2D = cell(nRat,1);
flipFlag1D  = false(nRat,1); flipFlag2D = false(nRat,1);
valRatsFold = cell(nRat,1);

for fold = 1:nRat
    rng(rngSeed+fold,"twister");

    idxTr = idxTrFold{fold};
    idxTe = idxTeFold{fold};

    if numel(idxTr)<50 || numel(idxTe)<5
        warning("Fold %d skipped (Ntr=%d, Nte=%d).", fold, numel(idxTr), numel(idxTe));
        continue;
    end

    fprintf("\n── Fold %d/%d | rat=%s | Ntr=%d Nte=%d ──\n", ...
        fold, nRat, char(uRat(fold)), numel(idxTr), numel(idxTe));

    Ytr_all   = Y(idxTr);
    ratTr_all = string(rat_id_all(idxTr));

    [trLocalIdx, valLocalIdx, valRats] = ...
        splitTrainValByRatStratified(Ytr_all, ratTr_all, nValRatsPerClass);
    valRatsFold{fold} = valRats;

    % ── 1D-CNN ──
    Xtr1 = Xcell(idxTr(trLocalIdx));   Ytr1 = Ytr_all(trLocalIdx);
    Xval1 = Xcell(idxTr(valLocalIdx)); Yval1 = Ytr_all(valLocalIdx);
    Xte1 = Xcell(idxTe);               Yte1 = Y(idxTe);

    [Xtr1, Ytr1] = balanceBinaryCellData(Xtr1, Ytr1);

    layers1D = [
        sequenceInputLayer(1,"MinLength",winN,"Name","seqIn")
        convolution1dLayer(11,16,"Padding","same","Name","conv1")
        batchNormalizationLayer("Name","bn1")
        reluLayer("Name","relu1")
        maxPooling1dLayer(2,"Stride",2,"Name","pool1")
        convolution1dLayer(9,32,"Padding","same","Name","conv2")
        batchNormalizationLayer("Name","bn2")
        reluLayer("Name","relu2")
        maxPooling1dLayer(2,"Stride",2,"Name","pool2")
        convolution1dLayer(7,64,"Padding","same","Name","conv3")
        batchNormalizationLayer("Name","bn3")
        reluLayer("Name","relu3")
        globalAveragePooling1dLayer("Name","gap")
        fullyConnectedLayer(numel(classes),"Name","fc")
        softmaxLayer("Name","sm")
        classificationLayer("Name","out")];

    opts1D = trainingOptions("adam", ...
        "InitialLearnRate",1e-3,"MaxEpochs",maxEpochs1D, ...
        "MiniBatchSize",miniBatch1D,"Shuffle","every-epoch", ...
        "ValidationData",{Xval1,Yval1}, ...
        "Verbose",false,"Plots","none","ExecutionEnvironment","auto");

    net1D = trainNetwork(Xtr1, Ytr1, layers1D, opts1D);

    mcaoIdx1 = find(net1D.Layers(end).Classes == "MCAO",1);
    [~, scVal1] = classify(net1D, Xval1, "MiniBatchSize",miniBatch1D);
    [flip1,~,~] = decideFlipFromVal_RatAUC(scVal1(:,mcaoIdx1), Yval1, ratTr_all(valLocalIdx));
    flipFlag1D(fold) = flip1;

    [~, scTe1] = classify(net1D, Xte1, "MiniBatchSize",miniBatch1D);
    pTe1 = scTe1(:,mcaoIdx1);
    if flip1, pTe1 = 1-pTe1; end
    foldScore1D{fold} = pTe1;
    foldYte1D{fold}   = Yte1;

    % ── 2D-CNN ──
    Xtr4_all = cat(4, Ximgs{idxTr});
    Xtr4  = Xtr4_all(:,:,:,trLocalIdx);   Ytr2  = Ytr_all(trLocalIdx);
    Xval4 = Xtr4_all(:,:,:,valLocalIdx);  Yval2 = Ytr_all(valLocalIdx);
    Xte4  = cat(4, Ximgs{idxTe});         Yte2  = Y(idxTe);

    [Xtr4, Ytr2] = balanceBinaryImageData(Xtr4, Ytr2);

    layers2D = [
        imageInputLayer([imgH imgW 1],"Name","imgIn")
        convolution2dLayer(3,16,"Padding","same","Name","conv1")
        batchNormalizationLayer("Name","bn1")
        reluLayer("Name","relu1")
        maxPooling2dLayer(2,"Stride",2,"Name","pool1")
        convolution2dLayer(3,32,"Padding","same","Name","conv2")
        batchNormalizationLayer("Name","bn2")
        reluLayer("Name","relu2")
        maxPooling2dLayer(2,"Stride",2,"Name","pool2")
        convolution2dLayer(3,64,"Padding","same","Name","conv3")
        batchNormalizationLayer("Name","bn3")
        reluLayer("Name","relu3")
        globalAveragePooling2dLayer("Name","gap")
        fullyConnectedLayer(numel(classes),"Name","fc")
        softmaxLayer("Name","sm")
        classificationLayer("Name","out")];

    opts2D = trainingOptions("adam", ...
        "InitialLearnRate",1e-3,"MaxEpochs",maxEpochs2D, ...
        "MiniBatchSize",miniBatch2D,"Shuffle","every-epoch", ...
        "ValidationData",{Xval4,Yval2}, ...
        "Verbose",false,"Plots","none","ExecutionEnvironment","auto");

    net2D = trainNetwork(Xtr4, Ytr2, layers2D, opts2D);

    mcaoIdx2 = find(net2D.Layers(end).Classes == "MCAO",1);
    [~, scVal2] = classify(net2D, Xval4, "MiniBatchSize",miniBatch2D);
    [flip2,~,~] = decideFlipFromVal_RatAUC(scVal2(:,mcaoIdx2), Yval2, ratTr_all(valLocalIdx));
    flipFlag2D(fold) = flip2;

    [~, scTe2] = classify(net2D, Xte4, "MiniBatchSize",miniBatch2D);
    pTe2 = scTe2(:,mcaoIdx2);
    if flip2, pTe2 = 1-pTe2; end
    foldScore2D{fold} = pTe2;
    foldYte2D{fold}   = Yte2;
end

%% ── 5) Rat-level池化 ──────────────────────────────────────────────────────
yRat  = categorical(strings(nRat,1), categories(Y));
pHat1 = nan(nRat,1);
pHat2 = nan(nRat,1);

for fold = 1:nRat
    if isempty(foldScore1D{fold}), continue; end
    yRat(fold)  = mode(foldYte1D{fold});
    pHat1(fold) = mean(double(foldScore1D{fold}),"omitnan");
    pHat2(fold) = mean(double(foldScore2D{fold}),"omitnan");
end

valid   = ~isnan(pHat1) & ~isnan(pHat2) & ~isundefined(yRat);
yRat_v  = yRat(valid);
p1_v    = clampProb(pHat1(valid));
p2_v    = clampProb(pHat2(valid));
rat_v   = uRat(valid);
posCat  = categorical("MCAO", categories(yRat_v));

[~,~,~,AUC_1D] = perfcurve(yRat_v, p1_v, posCat);
[~,~,~,AUC_2D] = perfcurve(yRat_v, p2_v, posCat);

%% ── 6) Stacking logistic fusion（rat-level）────────────────────────
posName  = "MCAO";
pFuseM   = nan(numel(yRat_v),1);

for i = 1:numel(yRat_v)
    trMask = true(numel(yRat_v),1);  trMask(i) = false;
    Xtrm   = [logit(p1_v(trMask)), logit(p2_v(trMask))];
    ytrm   = yRat_v(trMask);

    % 若訓練集只有一個類別，回退為平均
    if numel(unique(ytrm)) < 2
        pFuseM(i) = mean([p1_v(i), p2_v(i)]);
        continue;
    end

    % 用 CV 判斷 meta-model 是否需要 flip
    CVM = fitclinear(Xtrm, ytrm, "Learner","logistic","Regularization","ridge", ...
        "Lambda",1e-3,"Solver","lbfgs","Leaveout","on","ClassNames",categories(yRat_v));
    [~, sc_cv] = kfoldPredict(CVM);
    pTr_cv = getProbFromScore(sc_cv, CVM.ClassNames, posName);

    mPos = mean(pTr_cv(ytrm==posCat),"omitnan");
    mNeg = mean(pTr_cv(ytrm~=posCat),"omitnan");
    flipMeta = isfinite(mPos) && isfinite(mNeg) && (mPos < mNeg);

    % 重新擬合完整訓練集並預測
    mdlM = fitclinear(Xtrm, ytrm, "Learner","logistic","Regularization","ridge", ...
        "Lambda",1e-3,"Solver","lbfgs","ClassNames",categories(yRat_v));
    pTe = getPosteriorProbBinary(mdlM, [logit(p1_v(i)), logit(p2_v(i))], posName);
    if flipMeta, pTe = 1-pTe; end
    pFuseM(i) = pTe;
end

pFuseM = clampProb(pFuseM);
[~,~,~,AUC_Fuse] = perfcurve(yRat_v, pFuseM, posCat);

%% ── 7) 最終結果摘要 ──────────────────────────────────────────────────────
fprintf("\n=== Results (rat-level LOSO AUC) ===\n");
fprintf("1D  AUC = %.3f\n", AUC_1D);
fprintf("2D  AUC = %.3f\n", AUC_2D);
fprintf("Fuse AUC = %.3f\n", AUC_Fuse);

fprintf("\nPer-rat:\n");
for i = 1:numel(rat_v)
    fprintf("  %s | %s | p1=%.3f p2=%.3f pFuse=%.3f\n", ...
        char(rat_v(i)), char(string(yRat_v(i))), p1_v(i), p2_v(i), pFuseM(i));
end

%% ── 8) Per-rat 融合分數圖 ────────────────────────────────────────────────
pf = pFuseM;  yy = yRat_v;
if mean(pf(yy=="MCAO"),"omitnan") < mean(pf(yy=="Sham"),"omitnan")
    pf = 1-pf;
end

ord = [find(yy=="Sham"); find(yy=="MCAO")];
pf2 = pf(ord);  yy2 = yy(ord);

fig = figure("Color","w","Visible","off");
hold on;
scatter(find(yy2=="Sham"), pf2(yy2=="Sham"), 55, "filled");
scatter(find(yy2=="MCAO"), pf2(yy2=="MCAO"), 55, "filled");

mxS = max(pf2(yy2=="Sham"),[],"omitnan");
mnM = min(pf2(yy2=="MCAO"),[],"omitnan");
hasThreshold = false;
if isfinite(mxS) && isfinite(mnM) && mxS < mnM
    yline((mxS+mnM)/2, "--");
    hasThreshold = true;
end

grid on; box on;
xlabel("Rat index (Sham → MCAO)");
ylabel("pFuse — P(MCAO)");
title("Per-rat fused score (OOF)");

if hasThreshold
    legend("Sham","MCAO","Threshold","Location","best");
else
    legend("Sham","MCAO","Location","best");
end

try, exportgraphics(fig, fusePlotPath, "Resolution",300);
catch, saveas(fig, fusePlotPath); end
close(fig);
fprintf("Figure saved: %s\n", fusePlotPath);

%% ── 9) 儲存完整結果 ──────────────────────────────────────────────────────
save(resMatPath, ...
    "windowsMatPath","uRat","idxTrFold","idxTeFold","keepTopRMSFrac", ...
    "foldScore1D","foldYte1D","foldScore2D","foldYte2D", ...
    "pHat1","pHat2","yRat","valid", ...
    "rat_v","yRat_v","p1_v","p2_v","pFuseM", ...
    "AUC_1D","AUC_2D","AUC_Fuse", ...
    "flipFlag1D","flipFlag2D","valRatsFold", ...
    "winSec","hopRatio","winN","hopN","fs", ...
    "useBandpass","bpHz","useNotch","notchesHz","doDetrend","doMedianCenter", ...
    "doClip99","doRectify","doNormPerWin", ...
    "useActivityGate","envMethod","envWinMs","gateK","minActiveFracInWin","gateFallbackKeepAll", ...
    "imgH","imgW","stftBandHz", ...
    "capPerRatTrain","capPerRatTest","maxEpochs1D","maxEpochs2D","miniBatch1D","miniBatch2D", ...
    "nValRatsPerClass","rngSeed","-v7.3");

fprintf("Saved: %s\n", resMatPath);

%% ═══════════════════════════ Local Functions ════════════════════════════

function [Xb, Yb] = balanceBinaryCellData(X, Y)
% 對 1D cell 資料做二元類別平衡（下採樣多數類）
    Xb = X;  Yb = Y;
    nSh = sum(Y=="Sham");  nMc = sum(Y=="MCAO");
    if min(nSh,nMc)>0 && max(nSh,nMc) > 1.5*min(nSh,nMc)
        m = min(nSh,nMc);
        iSh = datasample(find(Y=="Sham"), m, "Replace",false);
        iMc = datasample(find(Y=="MCAO"), m, "Replace",false);
        keep = [iSh;iMc]; keep = keep(randperm(numel(keep)));
        Xb = X(keep);  Yb = Y(keep);
    end
end

function [Xb, Yb] = balanceBinaryImageData(X, Y)
% 對 4D image 資料做二元類別平衡
    Xb = X;  Yb = Y;
    nSh = sum(Y=="Sham");  nMc = sum(Y=="MCAO");
    if min(nSh,nMc)>0 && max(nSh,nMc) > 1.5*min(nSh,nMc)
        m = min(nSh,nMc);
        iSh = datasample(find(Y=="Sham"), m, "Replace",false);
        iMc = datasample(find(Y=="MCAO"), m, "Replace",false);
        keep = [iSh;iMc]; keep = keep(randperm(numel(keep)));
        Xb = X(:,:,:,keep);  Yb = Y(keep);
    end
end

function [trIdx, valIdx, valRats] = splitTrainValByRatStratified(Yall, ratAll, nValPerClass)
% 依老鼠做分層切割，各類別各取 nValPerClass 隻老鼠做 validation
    ratAll = string(ratAll(:));  Yall = categorical(Yall);
    u = unique(ratAll,"stable");
    yRat = categorical(strings(numel(u),1), categories(Yall));
    for i = 1:numel(u)
        yRat(i) = mode(Yall(ratAll==u(i)));
    end
    shamRats = u(yRat=="Sham");  mcaoRats = u(yRat=="MCAO");

    if numel(shamRats)<2 || numel(mcaoRats)<2
        rp = randperm(numel(Yall));
        nVal = max(10, round(0.1*numel(Yall)));
        valIdx = rp(1:nVal)';  trIdx = rp(nVal+1:end)';
        valRats = unique(ratAll(valIdx),"stable");
        return;
    end

    kS = min(nValPerClass, numel(shamRats)-1);
    kM = min(nValPerClass, numel(mcaoRats)-1);
    valRats = [shamRats(randperm(numel(shamRats),kS)); mcaoRats(randperm(numel(mcaoRats),kM))];

    valMask = ismember(ratAll, valRats);
    valIdx  = find(valMask);  trIdx = find(~valMask);

    if numel(unique(Yall(valIdx))) < 2
        valRats = [shamRats(randperm(numel(shamRats),1)); mcaoRats(randperm(numel(mcaoRats),1))];
        valMask = ismember(ratAll, valRats);
        valIdx  = find(valMask);  trIdx = find(~valMask);
    end
end

function [flip, aucP, aucF, dbg] = decideFlipFromVal_RatAUC(pVal_win, Yval_win, ratVal_win)
% 根據 validation rat-level AUC 決定是否翻轉機率輸出
    flip = false;  aucP = NaN;  aucF = NaN;
    dbg  = struct("nRatsValid",0,"nSham",0,"nMCAO",0);
    if isempty(pVal_win) || numel(unique(Yval_win))<2, return; end

    u = unique(string(ratVal_win(:)),"stable");
    pRat = nan(numel(u),1);
    yRat = categorical(strings(numel(u),1), categories(Yval_win));
    for i = 1:numel(u)
        m = (string(ratVal_win(:))==u(i));
        if ~any(m), continue; end
        pRat(i) = mean(double(pVal_win(m)),"omitnan");
        yRat(i) = mode(Yval_win(m));
    end

    valid = isfinite(pRat) & ~isundefined(yRat);
    if nnz(valid)<4 || numel(unique(yRat(valid)))<2, return; end

    dbg.nRatsValid = nnz(valid);
    dbg.nSham = sum(yRat(valid)=="Sham");
    dbg.nMCAO = sum(yRat(valid)=="MCAO");
    if dbg.nSham<2 || dbg.nMCAO<2, return; end

    pos = categorical("MCAO", categories(yRat));
    [~,~,~,aucP] = perfcurve(yRat(valid), pRat(valid), pos);
    [~,~,~,aucF] = perfcurve(yRat(valid), 1-pRat(valid), pos);
    if isfinite(aucP) && isfinite(aucF) && aucF>aucP, flip = true; end
end

function [env, actMask, thr] = buildActivityMask(x, fs, envMethod, envWinMs, gateK)
% 計算訊號包絡並產生活動遮罩
    x = double(x(:));
    w = max(3, round(envWinMs*1e-3*fs));
    if mod(w,2)==0, w=w+1; end

    if lower(string(envMethod))=="rms"
        env = sqrt(movmean(x.^2, w));
    else  % arv
        env = movmean(abs(x), w);
    end

    medv = median(env,"omitnan");
    madv = median(abs(env-medv),"omitnan");
    if ~isfinite(madv) || madv<1e-12
        thr = prctile(env,75);
    else
        thr = medv + gateK*madv;
    end
    actMask = (env > thr);
end

function mask = applyCapPerRatByScore(maskIn, ratVec, capPerRat, scoreVec)
% 每隻老鼠依 score 高低保留至多 capPerRat 個窗口
    if capPerRat<=0, mask = maskIn; return; end
    idxTrue = find(maskIn);
    rr = string(ratVec(maskIn));
    mask = false(size(maskIn));
    for i = 1:numel(unique(rr,"stable"))
        u = unique(rr,"stable");
        ix = idxTrue(rr==u(i));
        if isempty(ix), continue; end
        if numel(ix)>capPerRat
            sc = double(scoreVec(ix));
            sc(~isfinite(sc)) = -Inf;
            [~,ord] = sort(sc,"descend");
            ix = ix(ord(1:capPerRat));
        end
        mask(ix) = true;
    end
end

function maskOut = applyTopFracRMS(maskIn, ratVec, rmsVec, topFrac)
% 每隻老鼠依 RMS 保留前 topFrac 比例的窗口
    if topFrac>=1.0, maskOut = maskIn; return; end
    maskOut = false(size(maskIn));
    idxTrue = find(maskIn);
    rr = string(ratVec(maskIn));
    u = unique(rr,"stable");
    for i = 1:numel(u)
        ix = idxTrue(rr==u(i));
        if isempty(ix), continue; end
        k = max(1, round(topFrac*numel(ix)));
        [~,ord] = sort(rmsVec(ix),"descend");
        maskOut(ix(ord(1:k))) = true;
    end
end

function p = clampProb(p)
    p = max(min(double(p), 1-1e-6), 1e-6);
end

function z = logit(p)
    z = log(clampProb(p) ./ (1-clampProb(p)));
end

function T = readLabelSheetForce(xlsxPath, sheetName)
% 從 Excel 讀取標記表，欄位需含 file_path, start_s/t0, end_s/t1
    C = readcell(xlsxPath,"Sheet",sheetName,"TextType","string");
    if isempty(C), T=table(); return; end

    rowKeep = ~all(cellfun(@isEmptyCell, C), 2);
    colKeep = ~all(cellfun(@isEmptyCell, C), 1);
    C = C(rowKeep, colKeep);
    if isempty(C), T=table(); return; end

    headers = string(C(1,:));
    headers(headers=="") = "var" + string(find(headers==""));
    headers = matlab.lang.makeValidName(headers,"ReplacementStyle","delete");
    Traw = cell2table(C(2:end,:), "VariableNames",cellstr(headers));
    for k = 1:width(Traw)
        if ~isstring(Traw.(k)), Traw.(k) = string(Traw.(k)); end
    end

    T = table();
    if ~ismember("file_path",Traw.Properties.VariableNames)
        error("Sheet %s missing: file_path", sheetName);
    end
    T.file_path = strtrim(Traw.file_path);

    if ismember("group",Traw.Properties.VariableNames)
        T.group = string(strtrim(Traw.group));
    else
        T.group = string(sheetName);
    end
    g = lower(T.group);
    T.group(contains(g,"sham")) = "Sham";
    T.group(contains(g,"mcao")) = "MCAO";

    if ismember("start_s",Traw.Properties.VariableNames)
        T.t0 = str2double(Traw.start_s);
    elseif ismember("t0",Traw.Properties.VariableNames)
        T.t0 = str2double(Traw.t0);
    else
        error("Sheet %s missing: start_s / t0", sheetName);
    end

    if ismember("end_s",Traw.Properties.VariableNames)
        T.t1 = str2double(Traw.end_s);
    elseif ismember("t1",Traw.Properties.VariableNames)
        T.t1 = str2double(Traw.t1);
    else
        error("Sheet %s missing: end_s / t1", sheetName);
    end

    T = T(strlength(T.file_path)>0 & ~isnan(T.t0) & ~isnan(T.t1) & T.t1>T.t0, :);
end

function tf = isEmptyCell(x)
    if ismissing(x), tf=true; return; end
    if isstring(x)||ischar(x), tf=(strlength(string(x))==0); return; end
    if isnumeric(x), tf=(isscalar(x)&&isnan(x)); return; end
    tf = false;
end

function [emg, fs] = read_emg_ch2(matPath, defaultFS)
% 讀取 .mat 檔，取第 2 channel 作為 EUS-EMG
    S = load(matPath);
    if     isfield(S,"data"), A = S.data;
    elseif isfield(S,"sig"),  A = S.sig;
    elseif isfield(S,"RAW"),  A = S.RAW;
    else,  error("Cannot find data/sig/RAW in %s", matPath);
    end
    if isstruct(A) && isfield(A,"values"), A = A.values; end
    A = double(A);
    if size(A,2)<2, error("Need ≥2 channels (ch1=pressure, ch2=EMG)."); end
    emg = A(:,2);  fs = defaultFS;
end

function rid = extractRatIdFromPath(file_path)
% 從路徑中擷取 rat ID
    fp = string(file_path);
    rid = strings(numel(fp),1);
    for i = 1:numel(fp)
        m = regexp(char(fp(i)), 'r\d{3,5}[a-z]?', 'match', 'once');
        rid(i) = string(m);
        if strlength(rid(i))==0, rid(i) = "<UNK>"; end
    end
end

function p = getPosteriorProbBinary(mdl, X, className)
% 從二元線性分類器取指定類別的後驗機率
    [~, sc] = predict(mdl, X);
    sc = squeeze(sc);
    if isvector(sc)&&numel(sc)==2, sc=reshape(sc,1,2); end
    idx = find(string(mdl.ClassNames)==string(className),1);
    if isempty(idx), error("Class %s not found.", className); end
    p = clampProb(sc(:,idx));
end

function p = getProbFromScore(score, classNames, className)
    sc = squeeze(score);
    if isvector(sc)&&numel(sc)==2, sc=reshape(sc,1,2); end
    idx = find(string(classNames)==string(className),1);
    if isempty(idx), error("Class %s not found.", className); end
    p = sc(:,idx);
end

function saveExampleFigures(Xcell, Ximgs, Y, scoreVec, fs, stftBandHz, outDir)
% 輸出 GitHub / README 用的示例圖片
% 每組（Sham / MCAO）各選一個代表性視窗：
% 優先選 activity score 較高者；若 score 無效則取該組第一個

    if ~isfolder(outDir)
        mkdir(outDir);
    end

    yStr = string(Y);
    groupsToSave = ["Sham","MCAO"];

    for gi = 1:numel(groupsToSave)
        grp = groupsToSave(gi);
        idx = find(yStr == grp);

        if isempty(idx)
            continue;
        end

        sc = double(scoreVec(idx));
        if all(~isfinite(sc))
            k = idx(1);
        else
            sc(~isfinite(sc)) = -Inf;
            [~, imx] = max(sc);
            k = idx(imx);
        end

        % ===== 1) 時域波形圖 =====
        x = double(Xcell{k}(:));
        t = (0:numel(x)-1) / fs;

        fig1 = figure("Color","w","Visible","off");
        plot(t, x, "k-", "LineWidth", 1);
        grid on; box on;
        xlabel("Time (s)");
        ylabel("Amplitude (a.u.)");
        title(sprintf("Example EUS-EMG window (%s)", char(grp)));

        waveformPath = fullfile(outDir, "example_waveform_" + grp + ".png");
        try
            exportgraphics(fig1, waveformPath, "Resolution", 300);
        catch
            saveas(fig1, waveformPath);
        end
        close(fig1);

        % ===== 2) STFT 影像圖 =====
        img = double(Ximgs{k}(:,:,1));

        fig2 = figure("Color","w","Visible","off");
        imagesc(img);
        axis image;
        set(gca, "YDir", "normal");
        colormap(parula);
        colorbar;
        xlabel("Time ");
        ylabel("Frequency bins");
        title(sprintf("Example resized STFT image for 2D CNN (%s, %d-%d Hz)", ...
    char(grp), stftBandHz(1), stftBandHz(2)));

        stftPath = fullfile(outDir, "example_stft_" + grp + ".png");
        try
            exportgraphics(fig2, stftPath, "Resolution", 300);
        catch
            saveas(fig2, stftPath);
        end
        close(fig2);
    end

    fprintf("Example figures saved to: %s\n", outDir);
end