clear; clc; close all;

%% ================== 使用者參數 ==================
labelFile  = "YOUR_LABEL_FILE.xlsx"; % 請依實際情況修改此路徑
defaultFS  = 5000;

% 前處理
useBandpass     = true;
bpHz            = [50 500];
useNotch        = true;
notchesHz       = [60 120 180];
doDetrend       = true;
doMedianCenter  = true;
doClip99        = false;   % 與主流程一致，預設 false
doRectify       = false;   % 與主流程一致，預設 false

% 頻帶定義
lowBandHz  = [50 100];
highBandHz = [100 300];

% 每個 burst 的頻譜法
% 用 pwelch 算 PSD，再對 band 積分
welchWinSec = 0.10;      % 100 ms
welchOvlp   = 0.50;      % 50% overlap
welchNfft   = 2048;

% 輸出
outDir = pwd;
if ~isfolder(outDir)
    mkdir(outDir);
end

timeTag = string(datetime("now","Format","yyyyMMdd_HHmmss"));
figPath = fullfile(outDir, "HighLowRatio_ratlevel_" + timeTag + ".png");
csvPath = fullfile(outDir, "HighLowRatio_ratlevel_" + timeTag + ".csv");

fprintf("=== plot_highlow_ratio_ratlevel | %s ===\n", char(timeTag));

%% ================== 1) 讀 labels ==================
Tsham = readLabelSheetForce(labelFile, "Sham");
Tmcao = readLabelSheetForce(labelFile, "MCAO");
T = [Tsham; Tmcao];

if isempty(T) || height(T) == 0
    error("labels 為空，請檢查：%s", labelFile);
end

T.fs(:) = defaultFS;
T.rat_id = extractRatIdFromPath(T.file_path);

% 去掉 rat_id 無法辨識者
T = T(T.rat_id ~= "<UNK>", :);

if height(T) == 0
    error("所有 rat_id 都無法解析，請檢查檔名是否含 rXXXX。");
end

fprintf("✔ labels loaded: %d bursts\n", height(T));

%% ================== 2) 對每個 burst 計算 high/low ratio ==================
burstRatio = nan(height(T),1);
burstHigh  = nan(height(T),1);
burstLow   = nan(height(T),1);
burstDur   = nan(height(T),1);

nUsed = 0;
nSkip = 0;

for i = 1:height(T)
    fp = string(T.file_path(i));
    t0 = double(T.t0(i));
    t1 = double(T.t1(i));

    if ~isfile(fp)
        warning("找不到檔案：%s，略過。", fp);
        nSkip = nSkip + 1;
        continue;
    end

    try
        [emg, fs] = read_emg_ch2(fp, defaultFS);
    catch ME
        warning("讀取失敗：%s | %s", fp, ME.message);
        nSkip = nSkip + 1;
        continue;
    end

    i0 = max(1, floor(t0*fs) + 1);
    i1 = min(numel(emg), floor(t1*fs));

    if i1 <= i0
        warning("burst 長度 <= 0：%s | t0=%.4f t1=%.4f", fp, t0, t1);
        nSkip = nSkip + 1;
        continue;
    end

    seg = double(emg(i0:i1));
    seg = seg - mean(seg, "omitnan");

    % ----- 與主流程一致的前處理 -----
    if useNotch
        for hz = notchesHz
            w = [hz-1 hz+1] / (fs/2);
            if any(w <= 0) || any(w >= 1)
                continue;
            end
            [bn, an] = butter(2, w, "stop");
            seg = filtfilt(bn, an, seg);
        end
    end

    if useBandpass
        [b, a] = butter(4, bpHz/(fs/2), "bandpass");
        seg = filtfilt(b, a, seg);
    end

    if doDetrend
        seg = detrend(seg);
    end

    if doMedianCenter
        seg = seg - median(seg, "omitnan");
    end

    if doClip99
        pv = prctile(abs(seg), 99);
        if pv > 0
            seg = max(min(seg, 3*pv), -3*pv);
        end
    end

    if doRectify
        seg = abs(seg);
    end

    % 至少要有一些長度才適合做 Welch
    if numel(seg) < max(128, round(welchWinSec*fs))
        warning("burst 太短，略過：%s", fp);
        nSkip = nSkip + 1;
        continue;
    end

    % ----- PSD -----
    wlen = max(64, round(welchWinSec * fs));
    if mod(wlen,2) == 1
        wlen = wlen + 1;
    end
    noverlap = round(wlen * welchOvlp);

    [Pxx, F] = pwelch(seg, hamming(wlen, "periodic"), noverlap, welchNfft, fs, "power");

    % band power（用 trapz 積分）
    lowMask  = (F >= lowBandHz(1))  & (F < lowBandHz(2));
    highMask = (F >= highBandHz(1)) & (F < highBandHz(2));

    lowPow  = trapz(F(lowMask),  Pxx(lowMask));
    highPow = trapz(F(highMask), Pxx(highMask));

    if ~isfinite(lowPow) || lowPow <= 0 || ~isfinite(highPow)
        warning("band power 無效，略過：%s", fp);
        nSkip = nSkip + 1;
        continue;
    end

    burstLow(i)   = lowPow;
    burstHigh(i)  = highPow;
    burstRatio(i) = highPow / lowPow;
    burstDur(i)   = (i1 - i0 + 1) / fs;

    nUsed = nUsed + 1;
end

fprintf("✔ bursts used = %d | skipped = %d\n", nUsed, nSkip);

% 只保留有效 burst
validBurst = isfinite(burstRatio) & isfinite(burstHigh) & isfinite(burstLow);
T = T(validBurst,:);
burstRatio = burstRatio(validBurst);
burstHigh  = burstHigh(validBurst);
burstLow   = burstLow(validBurst);
burstDur   = burstDur(validBurst);

if isempty(burstRatio)
    error("沒有有效 burst ratio，可檢查頻帶或前處理設定。");
end

%% ================== 3) 聚合成 rat-level ==================
uRat = unique(T.rat_id, "stable");
nRat = numel(uRat);

ratRatio = nan(nRat,1);
ratGroup = strings(nRat,1);
ratNburst = zeros(nRat,1);
ratHigh = nan(nRat,1);
ratLow  = nan(nRat,1);

for r = 1:nRat
    m = (T.rat_id == uRat(r));

    ratRatio(r)  = mean(burstRatio(m), "omitnan");
    ratHigh(r)   = mean(burstHigh(m),  "omitnan");
    ratLow(r)    = mean(burstLow(m),   "omitnan");
    ratNburst(r) = sum(m);

    % 同一隻 rat 應該只屬於單一 group
    gg = string(T.group(find(m,1,"first")));
    if strlength(gg) == 0
        gg = "Unknown";
    end
    ratGroup(r) = gg;
end

% 轉成 categorical，並移除未用類別 / 排序
ratGroup = categorical(ratGroup);
ratGroup = removecats(ratGroup);  

wantOrder = ["MCAO","Sham"];
haveCats = string(categories(ratGroup, OutputType="string"));
keepOrder = wantOrder(ismember(wantOrder, haveCats));

if ~isempty(keepOrder)
    ratGroup = reordercats(ratGroup, keepOrder);  
end

% 再建立 rat index 1..N
ratIndex = (1:nRat)';

% 匯出表格
ratTable = table( ...
    ratIndex, string(uRat), string(ratGroup), ratRatio, ratHigh, ratLow, ratNburst, ...
    'VariableNames', {'rat_index','rat_id','group','high_low_ratio','mean_high_power','mean_low_power','n_bursts'} ...
);

writetable(ratTable, csvPath);
fprintf("📄 Saved rat-level CSV: %s\n", csvPath);

%% ================== 4) 畫圖 ==================
fig = figure('Color','w','Position',[100 100 1100 450]);
tl = tiledlayout(1,2, 'TileSpacing','compact', 'Padding','compact');

% ---------- 左圖：Group boxchart ----------
nexttile;
hold on;

% 用 boxchart 畫 group box plot
boxchart(ratGroup, ratRatio, 'BoxFaceColor', [0.30 0.60 0.90], 'MarkerStyle', 'o');
ylabel('High/Low ratio (100–300 Hz / 50–100 Hz)');
title('High/Low Ratio by Group (rat-level)');
grid on;
box on;

% x 軸只應該有 MCAO / Sham
ax = gca;
ax.FontSize = 11;

% ---------- 右圖：Per-rat scatter with rat index ----------
nexttile;
hold on;

isMCAO = (ratGroup == 'MCAO');
isSham = (ratGroup == 'Sham');

h1 = scatter(ratIndex(isMCAO), ratRatio(isMCAO), 42, 'filled', ...
    'MarkerFaceColor', [0.00 0.45 0.74], 'MarkerEdgeColor', 'k');
h2 = scatter(ratIndex(isSham), ratRatio(isSham), 42, 'filled', ...
    'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k');

xlabel('Rat index');
ylabel('High/Low ratio (100–300 Hz / 50–100 Hz)');
title('High/Low Ratio per Rat');
xlim([0.5, nRat + 0.5]);
xticks(1:2:nRat);
grid on;
box on;
legend([h1 h2], {'MCAO','Sham'}, 'Location','best');


% 存圖
try
    exportgraphics(fig, figPath, 'Resolution', 300);  
catch
    saveas(fig, figPath);
end

fprintf("🖼️ Saved figure: %s\n", figPath);

%% ================== 5) 基本統計輸出 ==================
fprintf("\n=== Summary ===\n");
fprintf("N rats = %d\n", nRat);
fprintf("MCAO   = %d\n", sum(ratGroup == 'MCAO'));
fprintf("Sham   = %d\n", sum(ratGroup == 'Sham'));

if sum(ratGroup == 'MCAO') > 0
    fprintf("MCAO mean ratio = %.4f\n", mean(ratRatio(ratGroup == 'MCAO'), 'omitnan'));
end
if sum(ratGroup == 'Sham') > 0
    fprintf("Sham mean ratio = %.4f\n", mean(ratRatio(ratGroup == 'Sham'), 'omitnan'));
end

fprintf("\nDone.\n");

%% ================== Local functions ==================

function T = readLabelSheetForce(xlsxPath, sheetName)
    C = readcell(xlsxPath, "Sheet", sheetName, "TextType","string");
    if isempty(C)
        T = table();
        return;
    end

    % 移除全空列
    rowKeep = false(size(C,1),1);
    for i = 1:size(C,1)
        rowKeep(i) = any(~cellfun(@(x) isEmptyCell(x), C(i,:)));
    end
    C = C(rowKeep,:);

    % 移除全空欄
    colKeep = false(1,size(C,2));
    for j = 1:size(C,2)
        colKeep(j) = any(~cellfun(@(x) isEmptyCell(x), C(:,j)));
    end
    C = C(:,colKeep);

    if isempty(C)
        T = table();
        return;
    end

    headers = string(C(1,:));
    emptyMask = (headers == "");
    headers(emptyMask) = "var" + string(find(emptyMask));
    headers = matlab.lang.makeValidName(headers, "ReplacementStyle","delete");

    data = C(2:end,:);
    Traw = cell2table(data, "VariableNames", cellstr(headers));

    for k = 1:width(Traw)
        if ~isstring(Traw.(k))
            Traw.(k) = string(Traw.(k));
        end
    end

    T = table();

    if ismember("file_path", Traw.Properties.VariableNames)
        T.file_path = strtrim(Traw.file_path);
    else
        error("工作表 %s 缺少欄位 file_path。", sheetName);
    end

    if ismember("group", Traw.Properties.VariableNames)
        T.group = string(strtrim(Traw.group));
    else
        T.group = string(sheetName);
    end

    g = lower(T.group);
    T.group(contains(g, "sham")) = "Sham";
    T.group(contains(g, "mcao")) = "MCAO";

    if ismember("start_s", Traw.Properties.VariableNames)
        T.t0 = str2double(Traw.start_s);
    elseif ismember("t0", Traw.Properties.VariableNames)
        T.t0 = str2double(Traw.t0);
    else
        error("工作表 %s 缺少 start_s / t0 欄位。", sheetName);
    end

    if ismember("end_s", Traw.Properties.VariableNames)
        T.t1 = str2double(Traw.end_s);
    elseif ismember("t1", Traw.Properties.VariableNames)
        T.t1 = str2double(Traw.t1);
    else
        error("工作表 %s 缺少 end_s / t1 欄位。", sheetName);
    end

    ok = strlength(T.file_path) > 0 & ~isnan(T.t0) & ~isnan(T.t1) & (T.t1 > T.t0);
    T = T(ok,:);
end

function tf = isEmptyCell(x)
    if ismissing(x)
        tf = true;
        return;
    end
    if isstring(x) || ischar(x)
        tf = (strlength(string(x)) == 0);
        return;
    end
    if isnumeric(x)
        tf = isscalar(x) && isnan(x);
        return;
    end
    tf = false;
end

function [emg, fs] = read_emg_ch2(matPath, defaultFS)
    S = load(matPath);

    if isfield(S, "data")
        A = S.data;
    elseif isfield(S, "sig")
        A = S.sig;
    elseif isfield(S, "RAW")
        A = S.RAW;
    else
        error("在 %s 找不到 data / sig / RAW 變數。", matPath);
    end

    if isstruct(A) && isfield(A, "values")
        A = A.values;
    end

    A = double(A);

    if size(A,2) < 2
        error("資料至少需要 2 通道（ch1 壓力, ch2 EMG）。");
    end

    emg = A(:,2);
    fs = defaultFS;
end

function rid = extractRatIdFromPath(file_path)
    fp = string(file_path);
    n = numel(fp);
    rid = strings(n,1);

    for i = 1:n
        m = regexp(char(fp(i)), 'r\d{3,5}[a-z]?', 'match', 'once');
        if ~isempty(m)
            rid(i) = string(m);
        else
            rid(i) = "<UNK>";
        end
    end
end