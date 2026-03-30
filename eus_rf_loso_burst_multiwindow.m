clear; clc; close all;

%% =====使用者參數=====
labelFile   = "YOUR_LABEL_FILE.xlsx"; % 請依實際情況修改此路徑
defaultFS   = 5000;

useBandpass = true;
bpHz        = [20 500];

hopRatio    = 0.5;
winList     = [0.1 0.2 0.3 0.5 0.7 1.0];

rng(42, "twister");

%% ===== 讀兩張表（Sham / MCAO），標準欄位對齊 =====
Tsham = readLabelSheetForce(labelFile, "Sham");
Tmiao = readLabelSheetForce(labelFile, "MCAO");
T = [Tsham; Tmiao];

if isempty(T) || height(T) == 0
    error("labels 讀取結果為空：請檢查活頁簿內容。");
end

T.fs(:) = defaultFS;

[~, fn, ext] = arrayfun(@fileparts, cellstr(T.file_path), "UniformOutput", false);
T.filename = string(fn) + string(ext);
T.rat_id   = extractRatIdFromPath(T.file_path);

T = T(T.rat_id ~= "<UNK>", :);
if height(T) == 0
    error("所有標註列的 rat_id 都解析失敗，請檢查路徑命名。");
end

if ~ismember("group", string(T.Properties.VariableNames))
    error("labels 中找不到 'group' 欄位。");
end

fprintf("讀入 %d 筆標註。\n", height(T));

%% ===== Burst 統計分布 =====
T.burstLen = T.t1 - T.t0;

fprintf("\nBurst 長度統計 (n=%d):\n", height(T));
fprintf("  min=%.2fs  Q1=%.2fs  med=%.2fs  Q3=%.2fs  max=%.2fs\n", ...
    min(T.burstLen), prctile(T.burstLen,25), median(T.burstLen), ...
    prctile(T.burstLen,75), max(T.burstLen));

%% ===== 結果表格 =====
SummaryResults = table('Size', [0 6], ...
    'VariableTypes', {'double','double','double','double','double','double'}, ...
    'VariableNames', {'winSec','globalAUC','meanAcc','stdAcc','meanSens','meanSpec'});

%% ===== 主迴圈：對每個視窗長度分別處理 =====
for w = 1:numel(winList)
    winSec = winList(w);

    fprintf("\n========== Window: %.1f s ==========\n", winSec);

    T_filtered = T(T.burstLen >= winSec, :);
    fprintf("Bursts (>= %.1fs): %d / %d\n", winSec, height(T_filtered), height(T));

    if height(T_filtered) == 0
        warning("No bursts pass filter for winSec=%.1f, skipping.", winSec);
        continue;
    end

    winN = round(winSec * defaultFS);
    hopN = round(winSec * hopRatio * defaultFS);

    winData = {};
    Y = strings(0, 1);
    R = strings(0, 1);

    for i = 1:height(T_filtered)
        fp  = T_filtered.file_path(i);
        grp = T_filtered.group(i);
        rid = T_filtered.rat_id(i);
        t0  = T_filtered.t0(i);
        t1  = T_filtered.t1(i);

        if ~isfile(fp)
            warning("File not found: %s", fp);
            continue;
        end

        try
            [emg, fs] = read_emg_ch2(fp, defaultFS);
        catch ME
            warning("Read failed (%s): %s", fp, ME.message);
            continue;
        end

        i0  = max(1, floor(t0*fs) + 1);
        i1  = min(numel(emg), floor(t1*fs));
        if i1 <= i0, continue; end

        seg = emg(i0:i1);
        seg = seg - mean(seg, 'omitnan');

        % Notch filters: 60/120/180 Hz
        for hz = [60 120 180]
            [bn, an] = butter(2, [hz-1 hz+1]/(fs/2), "stop");
            seg = filtfilt(bn, an, seg);
        end

        % Bandpass
        if useBandpass
            [b, a] = butter(4, bpHz/(fs/2), "bandpass");
            seg = filtfilt(b, a, seg);
        end

        seg = detrend(seg);
        seg = seg - median(seg);

        % Clip at 99th percentile x3
        pv = prctile(abs(seg), 99);
        if pv > 0
            seg = max(min(seg, 3*pv), -3*pv);
        end

        % Sliding window
        nSamp = numel(seg);
        for idx = 1:hopN:(nSamp - winN + 1)
            iEnd = idx + winN - 1;
            if iEnd > nSamp, break; end
            winData{end+1, 1} = seg(idx:iEnd);
            Y(end+1, 1) = grp;
            R(end+1, 1) = rid;
        end
    end

    fprintf("Total windows: %d\n", numel(winData));
    if isempty(winData)
        warning("No windows generated for winSec=%.1f, skipping.", winSec);
        continue;
    end

    labelsRF = double(Y == "MCAO");
    rat_idRF = R;

    fprintf("Sham: %d (%.1f%%)  MCAO: %d (%.1f%%)\n", ...
        sum(labelsRF==0), 100*mean(labelsRF==0), ...
        sum(labelsRF==1), 100*mean(labelsRF==1));

    [globalAUC, meanAcc, stdAcc, meanSens, meanSpec] = ...
        eus_run_rf_loso_emg(winData, labelsRF, rat_idRF, defaultFS, winSec);

    SummaryResults = [SummaryResults; ...
        {winSec, globalAUC, meanAcc, stdAcc, meanSens, meanSpec}];
end

%% ===== 結果比較 =====
fprintf("\n========== Results ==========\n");
disp(SummaryResults);

if height(SummaryResults) > 1
    [~, bestIdx] = max(SummaryResults.globalAUC);
    fprintf("Best window: %.1fs  AUC=%.3f  Acc=%.3f±%.3f  Sens=%.3f  Spec=%.3f\n", ...
        SummaryResults.winSec(bestIdx), SummaryResults.globalAUC(bestIdx), ...
        SummaryResults.meanAcc(bestIdx), SummaryResults.stdAcc(bestIdx), ...
        SummaryResults.meanSens(bestIdx), SummaryResults.meanSpec(bestIdx));

    figure;
    metrics = {'globalAUC','meanAcc','meanSens','meanSpec'};
    titles  = {'Global AUC','Mean Acc','Mean Sens','Mean Spec'};
    for m = 1:4
        subplot(2,2,m);
        bar(SummaryResults.winSec, SummaryResults.(metrics{m}));
        xlabel('Window (s)'); ylabel(titles{m});
        title(titles{m}); grid on;
    end
end


%% ===== Local functions =====

function [globalAUC, meanAcc, stdAcc, meanSens, meanSpec] = ...
    eus_run_rf_loso_emg(winData, labels, rat_id, fs, winSec)

fprintf('\n[RF %.1fs] Extracting features...\n', winSec);

Nwin        = numel(winData);
feat_sample = extract_emg_features(winData{1}, fs);
D           = numel(feat_sample);

X_feat = zeros(Nwin, D);
for i = 1:Nwin
    X_feat(i, :) = extract_emg_features(winData{i}, fs);
end
fprintf('[RF %.1fs] %d windows x %d features.\n', winSec, Nwin, D);

rng(0);
rat_id = string(rat_id(:));
labels = labels(:);
rats   = unique(rat_id);
Nrats  = numel(rats);

acc_per_rat  = zeros(Nrats, 1);
sens_per_rat = zeros(Nrats, 1);
spec_per_rat = zeros(Nrats, 1);
all_scores   = [];
all_ytrue    = [];

for k = 1:Nrats
    isTest  = (rat_id == rats(k));
    isTrain = ~isTest;

    y_train_cat = categorical(labels(isTrain));
    y_test      = labels(isTest);
    y_test_cat  = categorical(y_test);

    t   = templateTree('MaxNumSplits', 20);
    Mdl = fitcensemble(X_feat(isTrain,:), y_train_cat, ...
        'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', t);

    [y_pred_cat, scores] = predict(Mdl, X_feat(isTest,:));

    posClass = categorical(1);
    idxPos   = find(Mdl.ClassNames == posClass);
    if isempty(idxPos)
        error('ClassNames does not contain class 1. Check labels.');
    end

    pos_scores = scores(:, idxPos);
    all_scores = [all_scores; pos_scores];
    all_ytrue  = [all_ytrue; y_test];

    y_pred_num = double(y_pred_cat == posClass);
    y_test_num = double(y_test_cat == posClass);

    TP = sum((y_test_num==1) & (y_pred_num==1));
    TN = sum((y_test_num==0) & (y_pred_num==0));
    FP = sum((y_test_num==0) & (y_pred_num==1));
    FN = sum((y_test_num==1) & (y_pred_num==0));

    acc_per_rat(k)  = (TP+TN) / max(TP+TN+FP+FN, 1);
    sens_per_rat(k) = ternary(TP+FN>0, TP/(TP+FN), NaN);
    spec_per_rat(k) = ternary(TN+FP>0, TN/(TN+FP), NaN);

    fprintf('[Fold %2d/%2d] %s | n=%4d | Acc=%.3f Sens=%.2f Spec=%.2f\n', ...
        k, Nrats, rats(k), sum(isTest), acc_per_rat(k), sens_per_rat(k), spec_per_rat(k));
end

% Global AUC: pick direction where AUC >= 0.5
if numel(unique(all_ytrue)) < 2
    warning('[RF %.1fs] Only one class in test set; AUC undefined.', winSec);
    globalAUC = NaN;
    fpr = []; tpr = [];
else
    [fpr1, tpr1, ~, auc1] = perfcurve(all_ytrue, all_scores,   1);
    [fpr2, tpr2, ~, auc2] = perfcurve(all_ytrue, 1-all_scores, 1);
    if auc2 > auc1
        globalAUC  = auc2; fpr = fpr2; tpr = tpr2;
        all_scores = 1 - all_scores;
    else
        globalAUC  = auc1; fpr = fpr1; tpr = tpr1;
    end
end

meanAcc  = mean(acc_per_rat,  'omitnan');
stdAcc   = std(acc_per_rat,   'omitnan');
meanSens = mean(sens_per_rat, 'omitnan');
meanSpec = mean(spec_per_rat, 'omitnan');

fprintf('\n[RF %.1fs] Global AUC=%.3f  Acc=%.3f±%.3f  Sens=%.3f  Spec=%.3f\n', ...
    winSec, globalAUC, meanAcc, stdAcc, meanSens, meanSpec);

if ~isnan(globalAUC)
    figure;
    plot(fpr, tpr, 'LineWidth', 1.5); hold on;
    plot([0 1],[0 1],'--');
    xlabel('FPR'); ylabel('TPR');
    title(sprintf('RF %.1fs  AUC=%.3f', winSec, globalAUC));
    grid on;
end
end


function featVec = extract_emg_features(x, fs)
x   = x(:);
L   = length(x);

% Time-domain
meanAbs = mean(abs(x));
rmsVal  = rms(x);
stdVal  = std(x);
zc      = sum(x(1:end-1) .* x(2:end) < 0) / max(L-1, 1);
wfl     = sum(abs(diff(x)));

% Frequency-domain (Welch PSD)
[pxx, f] = pwelch(x, [], [], [], fs);

P_total   = bandpower(pxx, f, [10  500], 'psd');
P_20_80   = bandpower(pxx, f, [20   80], 'psd');
P_80_150  = bandpower(pxx, f, [80  150], 'psd');
P_150_300 = bandpower(pxx, f, [150 300], 'psd');

mask  = (f >= 10) & (f <= 500);
P_seg = pxx(mask);
f_seg = f(mask);

if sum(P_seg) > 0
    specCentroid = sum(f_seg .* P_seg) / sum(P_seg);
    cumP         = cumsum(P_seg);
    medFreq      = f_seg(find(cumP >= cumP(end)/2, 1));
else
    specCentroid = 0;
    medFreq      = 0;
end

featVec = [meanAbs, rmsVal, stdVal, zc, wfl, ...
           P_total, P_20_80, P_80_150, P_150_300, ...
           specCentroid, medFreq];
end


function T = readLabelSheetForce(xlsxPath, sheetName)
C = readcell(xlsxPath, "Sheet", sheetName, "TextType", "string");
if isempty(C), T = table(); return; end

rowKeep = ~all(cellfun(@isEmptyCell, C), 2);
colKeep = ~all(cellfun(@isEmptyCell, C), 1);
C = C(rowKeep, colKeep);
if isempty(C), T = table(); return; end

headers = string(C(1,:));
headers(headers=="") = "var" + string(find(headers==""));
headers = matlab.lang.makeValidName(headers, "ReplacementStyle", "delete");
Traw = cell2table(C(2:end,:), "VariableNames", cellstr(headers));
for k = 1:width(Traw)
    if ~isstring(Traw.(k)), Traw.(k) = string(Traw.(k)); end
end

T = table();
if ~ismember("file_path", Traw.Properties.VariableNames)
    error("Sheet '%s' missing column: file_path", sheetName);
end
T.file_path = strtrim(Traw.file_path);

T.group = string(sheetName);
if ismember("group", Traw.Properties.VariableNames)
    T.group = string(strtrim(Traw.group));
end
g = lower(T.group);
T.group(contains(g,"sham")) = "Sham";
T.group(contains(g,"mcao")) = "MCAO";

T.fs = repmat(5000, height(Traw), 1);
if ismember("fs", Traw.Properties.VariableNames)
    T.fs = str2double(Traw.fs);
end

if ismember("start_s", Traw.Properties.VariableNames)
    T.t0 = str2double(Traw.start_s);
elseif ismember("t0", Traw.Properties.VariableNames)
    T.t0 = str2double(Traw.t0);
else
    error("Sheet '%s' missing column: start_s or t0", sheetName);
end

if ismember("end_s", Traw.Properties.VariableNames)
    T.t1 = str2double(Traw.end_s);
elseif ismember("t1", Traw.Properties.VariableNames)
    T.t1 = str2double(Traw.t1);
else
    error("Sheet '%s' missing column: end_s or t1", sheetName);
end

T = T(strlength(T.file_path)>0 & ~isnan(T.t0) & ~isnan(T.t1) & T.t1>T.t0, :);
end


function tf = isEmptyCell(x)
if ismissing(x),                           tf = true;  return; end
if isstring(x) || ischar(x),              tf = strlength(string(x)) == 0; return; end
if isnumeric(x) && isscalar(x) && isnan(x), tf = true; return; end
tf = false;
end


function [emg, fs] = read_emg_ch2(matPath, defaultFS)
S = load(matPath);
if      isfield(S,"data"), A = S.data;
elseif  isfield(S,"sig"),  A = S.sig;
elseif  isfield(S,"RAW"),  A = S.RAW;
else, error("No data/sig/RAW variable in %s", matPath);
end
if isstruct(A) && isfield(A,"values"), A = A.values; end
A = double(A);
if size(A,2) < 2, error("Need at least 2 channels (ch1=pressure, ch2=EMG)."); end
emg = A(:,2);
fs  = defaultFS;
end


function rid = extractRatIdFromPath(file_path)
fp  = string(file_path);
rid = strings(numel(fp), 1);
for i = 1:numel(fp)
    m = regexp(char(fp(i)), 'r\d{3,5}[a-z]?', 'match', 'once');
    rid(i) = string(ternary(~isempty(m), m, "<UNK>"));
end
end


function out = ternary(cond, a, b)
% Inline conditional: out = cond ? a : b
if cond, out = a; else, out = b; end
end