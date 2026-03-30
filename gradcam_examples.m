clear; clc; close all;

packFile = "YOUR_GRADCAM_PACK_FILE.mat"; % 請依實際情況修改此路徑

if ~isfile(packFile)
    error("找不到 GradCAM pack 檔案：%s\n請先由主分析流程產生對應的 pack 檔，或修改 packFile 為正確路徑。", packFile);
end

outDir = fullfile(pwd, "GradCAM_outputs");
if ~isfolder(outDir)
    mkdir(outDir);
end

S = load(packFile, "pack");
pack = S.pack;

fs   = double(pack.fs);
net1D = pack.net1D;
net2D = pack.net2D;

% class index for MCAO
cls1 = string(net1D.Layers(end).Classes);
cls2 = string(net2D.Layers(end).Classes);
idxM1 = find(cls1=="MCAO", 1);
idxM2 = find(cls2=="MCAO", 1);
if isempty(idxM1) || isempty(idxM2)
    error("Cannot find class MCAO in net classes.");
end

% Feature layers
feat1 = string(pack.gradcamConv1DLayer);
feat2 = string(pack.gradcamConv2DLayer);

% 想看第幾個 HIGH / LOW（1~10）
kHigh = 4;
kLow  = 4;

uidHigh = string(pack.high_uid(min(kHigh, numel(pack.high_uid))));
uidLow  = string(pack.low_uid(min(kLow,  numel(pack.low_uid))));

iHigh = find(string(pack.window_uid)==uidHigh, 1);
iLow  = find(string(pack.window_uid)==uidLow, 1);
if isempty(iHigh) || isempty(iLow)
    error("Cannot locate high/low uid inside pack.window_uid. Check pack fields.");
end

% Run for both
exportOne(pack, net1D, net2D, fs, idxM1, idxM2, feat1, feat2, iHigh, "HIGH", outDir);
exportOne(pack, net1D, net2D, fs, idxM1, idxM2, feat1, feat2, iLow,  "LOW",  outDir);

fprintf("DONE. Output: %s\n", outDir);

function exportOne(pack, net1D, net2D, fs, idxM1, idxM2, feat1, feat2, idx, tag, outDir)

    uid  = string(pack.window_uid(idx));
    y    = string(pack.Y(idx));
    rat  = string(pack.testRat);
    fold = double(pack.fold);

    x   = double(pack.Xcell_1d{idx}(:));            % winN x 1
    img = double(pack.Ximgs_2d{idx}(:,:,1));        % imgH x imgW

    % ---- sanity check ----
    [lab1, sc1] = classify(net1D, x', "MiniBatchSize", 1);
    pM1 = sc1(1, idxM1);
    fprintf("%s fold=%02d %s uid=%s | y=%s | pred1D=%s | p1D(MCAO)=%.4f\n", ...
        rat, fold, tag, uid, y, string(lab1), pM1);

    [lab2, sc2] = classify(net2D, img, "MiniBatchSize", 1);
    pM2 = sc2(1, idxM2);
    fprintf("    pred2D=%s | p2D(MCAO)=%.4f\n", string(lab2), pM2);

    % ---- 1D Grad-CAM ----
    cam1 = gradCAM(net1D, x', idxM1, "FeatureLayer", feat1);
    if iscell(cam1), cam1 = cam1{1}; end
    cam1 = double(cam1(:));

    lo1 = prctile(cam1, 5);
    hi1 = prctile(cam1, 99);
    cam1 = (cam1 - lo1) / max(hi1 - lo1, eps);
    cam1 = max(min(cam1, 1), 0);

    % ---- 2D Grad-CAM ----
    cam2 = gradCAM(net2D, img, idxM2, "FeatureLayer", feat2);
    cam2 = double(cam2);

    lo2 = prctile(cam2(:), 5);
    hi2 = prctile(cam2(:), 99);
    cam2 = (cam2 - lo2) / max(hi2 - lo2, eps);
    cam2 = max(min(cam2, 1), 0);

    % ===== 軸標籤 =====
    t = (0:numel(x)-1)/fs;
    imgH = size(img,1);
    imgW = size(img,2);

    time_ms = 0:50:300;
    xt = 1 + round((time_ms/(pack.winSec*1000)) * (imgW-1));
    xt = min(max(xt,1), imgW);

    freq_hz = [20 100 200 300 400 500];
    yt = 1 + round(((freq_hz - pack.stftBandHz(1)) / ...
         (pack.stftBandHz(2)-pack.stftBandHz(1))) * (imgH-1));
    yt = min(max(yt,1), imgH);

    % ===== 建立 2D overlay：灰階底圖 + 彩色 CAM =====
    % 底圖先正規化到 0~1
    imgN = img - min(img(:));
    imgN = imgN / max(max(imgN(:)), eps);

    % 灰階底圖轉 RGB
    imgRGB = repmat(imgN, [1 1 3]);

    % CAM 轉成 RGB 
    cmap = jet(256);
    camIdx = max(1, min(256, round(cam2*255)+1));
    camRGB = ind2rgb(camIdx, cmap);

    % 只顯示高 CAM 區
    camThr = 0.60;   
    alphaMap = zeros(size(cam2));
    alphaMap(cam2 >= camThr) = 0.80 * cam2(cam2 >= camThr);

    % alpha blending
    overlayRGB = imgRGB;
    for c = 1:3
        overlayRGB(:,:,c) = (1-alphaMap).*imgRGB(:,:,c) + alphaMap.*camRGB(:,:,c);
    end

    % ===== plot =====
    f = figure("Color","w","Visible","off","Position",[60 60 1450 760]);
    tiledlayout(2,3,"TileSpacing","compact","Padding","compact");

    % 1D waveform
    nexttile;
    plot(t*1000, x, "LineWidth", 0.8);
    grid on; box on;
    xlabel("Time (ms)"); ylabel("EMG");
    title(sprintf("%s | fold=%02d | %s | y=%s", rat, fold, tag, y), "Interpreter","none");

    % 1D Grad-CAM
    nexttile;
    plot(t*1000, cam1, "LineWidth", 1.0);
    ylim([0 1.05]);
    grid on; box on;
    xlabel("Time (ms)"); ylabel("Grad-CAM");
    title(sprintf("1D Grad-CAM | %s | class=MCAO", feat1), "Interpreter","none");

    % 1D 疊在波形上
    nexttile;
    yyaxis left;
    plot(t*1000, x, "LineWidth", 0.8); ylabel("EMG");
    yyaxis right;
    plot(t*1000, cam1, "LineWidth", 1.0); ylabel("Grad-CAM");
    grid on; box on;
    xlabel("Time (ms)");
    title("1D waveform + Grad-CAM");

    % 2D input
    nexttile;
    imagesc(img);
    axis xy; axis image; axis tight;
    set(gca, "XTick", xt, "XTickLabel", time_ms, ...
             "YTick", yt, "YTickLabel", freq_hz);
    xlabel("Time (ms)"); ylabel("Frequency (Hz)");
    colormap(gca, parula);
    colorbar;
    title("2D input (STFT image)");

    % 2D CAM heatmap 單獨畫
    nexttile;
    imagesc(cam2, [0 1]);
    axis xy; axis image; axis tight;
    set(gca, "XTick", xt, "XTickLabel", time_ms, ...
             "YTick", yt, "YTickLabel", freq_hz);
    xlabel("Time (ms)"); ylabel("Frequency (Hz)");
    colormap(gca, hot);
    colorbar;
    title(sprintf("2D Grad-CAM heatmap | %s", feat2), "Interpreter","none");

    % 2D masked overlay
    nexttile;
    image(overlayRGB);
    axis xy; axis image; axis tight;
    set(gca, "XTick", xt, "XTickLabel", time_ms, ...
             "YTick", yt, "YTickLabel", freq_hz);
    xlabel("Time (ms)"); ylabel("Frequency (Hz)");
    title(sprintf("2D Grad-CAM overlay (thr=%.2f)", camThr));

    sgtitle(sprintf("%s | uid=%s | pred1D=%.3f | pred2D=%.3f", ...
        tag, uid, pM1, pM2), "Interpreter","none");

    outPng = fullfile(outDir, sprintf("%s_fold%02d_%s_uid_%s.png", ...
        rat, fold, tag, sanitize(uid)));
    exportgraphics(f, outPng, "Resolution", 220);
    close(f);
end
function s = sanitize(uid)
    s = char(uid);
    s = regexprep(s, '[^\w\-]', '_');
    if numel(s) > 140
        s = s(1:140);
    end
end