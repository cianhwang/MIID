clc;clear;close all

addpath ('../toolkit', '../toolkit/spec2rgb', '../toolkit/load_data')
nameList = {'ali','green_pig','mask', 'piggy_crown', 'spongebob','pumpkin', 'dinosaur', 'piggy_horse', 'hellokitty', 'cap', 'girl', 'fox'};
gpu = gpuDevice(1);

for nameIdx = 1:length(nameList)
    FILENAME = char(nameList(nameIdx));
    addpath(strcat('../miid_dataset/',FILENAME));

%% read spectral data
[specImg specImgRGB height width numSpec wl]=readDat('diffuse.dat');specImg = specImg(:, :, 1:3:119);
[gt_shad gt_shadRGB height width numSpec wl]=readDat('shading.dat');gt_shad = gt_shad(:, :, 1:3:119);

gt_refl =  specImg./(gt_shad+eps); 
mask = imread('mask.bmp');
mask = mean(mask, 3);
mask = logical(mask);
%groundtruth;

% specImgRGB = im2double(imread('diffuse.bmp'));
% figure, imshow(specImgRGB)
%---------------block--------------

[mask, specImg, gt_shad, gt_refl] = RESIZE(0.5^floor(nameIdx/6), mask, specImg, gt_shad, gt_refl);
[Row, Col, Chn] = size(specImg);
K = Chn;
N = Row*Col;

specImg = specImg.*repmat(mask, [1 1 K]);
specImgVec = reshape(specImg, [], K);
specImg = (specImg - min(specImg(:)))/(max(specImg(:))- min(specImg(:)));
gt_shad = (gt_shad - min(gt_shad(:)))/(max(gt_shad(:))- min(gt_shad(:)));

RGB = spec2rgb(specImg);
% figure, imshow(RGB); 

mImgRGB = RGB;
mImgRGB = im2double(mImgRGB);
mImg =specImg;

    addpath .\SLICtoolkit
    %% Parameters
    tic
	CHROM_TH = 0.9871;
    SHADING_SCALE = 0.5;
    SCALE_CON_SEG_SIZE = 30;
    % For SLICmShadImg
    SLIC_CLUSTER_NUM = 5000;
    SUPER_PIXEL_RATIO = 8;
    SE_RADIUS = 1;
    SUPER_PIXEL_KMEANS_NUM = 20;
    %% Compute super pixel image
    mImg(mImg == 0) = 1;
    mImg = im2double(mImg);
    [sImgHeight,sImgWidth,sImgChan] = size(mImg);
    [mLabel,mAdjacent] = slic(mImgRGB,SLIC_CLUSTER_NUM,SUPER_PIXEL_RATIO,SE_RADIUS);
    toc
    vLabel = reshape(mLabel,[sImgHeight*sImgWidth,1]);
    vImg = reshape(mImg,[sImgHeight*sImgWidth,sImgChan]);
    sSupPixNum = size(mAdjacent,1);
    vSupPixImg = zeros(sSupPixNum,sImgChan);
    for i = 1:sSupPixNum
        indice = find(vLabel == i);
        vSupPixImg(i,:) = mean(vImg(indice,:),1);
    end
    vSupPixImgL2Norm = GetL2Norm(vSupPixImg);
    %vSupPixClusterMap = kmeans(vSupPixImgL2Norm,SUPER_PIXEL_KMEANS_NUM,'distance','cosine');
    vSupPixClusterMap = SppClusterGen(1:sSupPixNum,vSupPixImgL2Norm,mAdjacent,SUPER_PIXEL_KMEANS_NUM);
    sSupPixTotChanNum = sSupPixNum;
    %% Compute Local Constrains
    % For Reflectance Constraint
    sRcEleSize = (max(size(find(mAdjacent == 1))))*sImgChan;
    sRcRowSize = sRcEleSize/2;% Reflectance Constraint Matrix element number
    vRcRowIdxs = zeros(sRcEleSize,1);% Constrains matrix row indice
    vRcColumnIdxs = zeros(sRcEleSize,1);% Constrains matrix column indice
    vRcEleValue = zeros(sRcEleSize,1);% Ref constrains matrix element values
    sRcRowCnt = 0;
    sRcEleCnt = 0;
    % For Shading Constraint
    sScEleSize = max(size(find(mAdjacent == 1)));
    sScRowSize = sScEleSize/2;
    vScRowIdxs = zeros(sScEleSize,1);
    vScColumnIdxs = zeros(sScEleSize,1);
    vScEleValue = zeros(sScEleSize,1);% Shading constrains matrix element values
    sScRowCnt = 0;
    sScEleCnt = 0;
    for i = 1:sSupPixNum
        for j = i:sSupPixNum
            if i>=j || mAdjacent(i,j) == 0
                continue;
            end
           if vSupPixImgL2Norm(i,:)*vSupPixImgL2Norm(j,:)' < CHROM_TH
               sScCoef = 1;
               sRcCoef = 0.1;
           else
               sRcCoef = 1;
               sScCoef = 0.5;
           end
           % Compute Shading Constraint value
           % For pixel p
           sScEleCnt = sScEleCnt+1;
           sScRowCnt = sScRowCnt+1;
           vScRowIdxs(sScEleCnt) = sScRowCnt;
           vScColumnIdxs(sScEleCnt) = j;
           vScEleValue(sScEleCnt) = sScCoef;
           % For pixel q
           sScEleCnt = sScEleCnt+1;
           vScRowIdxs(sScEleCnt) = sScRowCnt;
           vScColumnIdxs(sScEleCnt) = i;
           vScEleValue(sScEleCnt) = -sScCoef;
           % Compute Reflectance Constraint value
           for k = 1:sImgChan
               % For pixel p
               sRcEleCnt = sRcEleCnt+1;
               vRcRowIdxs(sRcEleCnt) = sRcRowCnt+k;
               vRcColumnIdxs(sRcEleCnt) = j;
               vRcEleValue(sRcEleCnt) = sRcCoef*vSupPixImg(i,k);
               % For pixel q
               sRcEleCnt = sRcEleCnt+1;
               vRcRowIdxs(sRcEleCnt) = sRcRowCnt+k;
               vRcColumnIdxs(sRcEleCnt) = i;
               vRcEleValue(sRcEleCnt) = -sRcCoef*vSupPixImg(j,k);               
           end
           sRcRowCnt = sRcRowCnt+sImgChan;
        end
    end
    mSc = sparse(vScRowIdxs,vScColumnIdxs,vScEleValue,sScRowSize,sSupPixTotChanNum,sScEleSize); mSc = gpuArray(mSc);
    mRc = sparse(vRcRowIdxs,vRcColumnIdxs,vRcEleValue,sRcRowSize,sSupPixTotChanNum,sRcEleSize); mRc = gpuArray(mRc);
    toc
    %% For Global Constrains
    sRowCnt = 0;   
    sEleCnt = 0;
    sEleNum = sSupPixNum*sImgChan*2;
    vRowIdxs = zeros(sEleNum,1);
    vColumnIdxs = zeros(sEleNum,1);
    vGcEleValue = zeros(sEleNum,1);
    for i = 1:SUPER_PIXEL_KMEANS_NUM
        sClusterIdxs = find(vSupPixClusterMap == i);
        sClusterSize = max(size(sClusterIdxs));
        if sClusterSize == 1
            continue;
        end
        for j = 1:sClusterSize
            while 1
                sRandiRes = randi(sClusterSize,1);
                if sRandiRes ~= j
                    break;
                end
            end
            for k = 1:sImgChan
                % For pixel p
                sEleCnt = sEleCnt+1;
                vRowIdxs(sEleCnt) = sRowCnt+k;
                vColumnIdxs(sEleCnt) = sClusterIdxs(j);
                vGcEleValue(sEleCnt) = vSupPixImg(sClusterIdxs(j),k);
                % For pixel q
                sEleCnt = sEleCnt+1;
                vRowIdxs(sEleCnt) = sRowCnt+k;           
                vColumnIdxs(sEleCnt) = sClusterIdxs(sRandiRes);
                vGcEleValue(sEleCnt) = -vSupPixImg(sClusterIdxs(sRandiRes),k);
            end
            sRowCnt = sRowCnt+sImgChan;
        end
    end
    mGc = sparse(vRowIdxs(1:sEleCnt),vColumnIdxs(1:sEleCnt),vGcEleValue(1:sEleCnt),sRowCnt,sSupPixTotChanNum,sEleNum);
    mGc = gpuArray(mGc);
    toc
    %% For scale constrains
    L = sparse(SCALE_CON_SEG_SIZE,sSupPixNum);
    for i = 1:SCALE_CON_SEG_SIZE-1
        L(i,floor(sSupPixNum/SCALE_CON_SEG_SIZE)*(i-1)+1:floor(sSupPixNum/SCALE_CON_SEG_SIZE)*i) = 1;
    end
    L(SCALE_CON_SEG_SIZE,floor(sSupPixNum/SCALE_CON_SEG_SIZE)*(SCALE_CON_SEG_SIZE-1)+1:sSupPixNum) = 1;
    b = ones(SCALE_CON_SEG_SIZE,1)*SHADING_SCALE*sSupPixNum/SCALE_CON_SEG_SIZE;
    b = L'*b;
    toc
    %% % Optimization
    A = (mSc'*mSc)*((sRowCnt+sRcRowCnt)/sScRowCnt)+mRc'*mRc+mGc'*mGc+L'*L;
    vSupPixShad = pcg(A,b,1e-6,100);
    if isempty(find(vSupPixShad < 0))
        disp('All reflectance values are positive!')
    else
        vSupPixShad = abs(vSupPixShad);
        disp('Negative reflectance value exist!');
    end
    vSupPixShad = reshape(vSupPixShad',[1,sSupPixNum]);
    vSupPixShad = vSupPixShad';
    vSupPixShad(vSupPixShad == 0) = 0.001;
    vShadImg = zeros(sImgHeight*sImgWidth,1);
    for i = 1:sSupPixNum
        vIdxs = find(vLabel == i);
        vShadImg(vIdxs,1) = gather(vSupPixShad(i,1));
    end
    mImg = mImg .* repmat(mask, [1 1 sImgChan]);
    mShadImg = reshape(vShadImg,[sImgHeight,sImgWidth,1]);
    mRefImg = mImg./(repmat(mShadImg,[1,1,sImgChan])+eps);
    for KKK = 1:sImgChan
     mRefImg(:, :, KKK) = BilateralFilt2(mRefImg(:, :, KKK),6,[3,0.1]);
    end
    %----------------------illumination-------------------
    Bs = [0.0304664857685566;0.0302258506417274;0.0301303472369909;0.0301258098334074;0.0304012447595596;0.0306360777467489;0.0310054197907448;0.0314002558588982;0.0319754667580128;0.0324994511902332;0.0331317000091076;0.0338466949760914;0.0344399325549603;0.0352818295359612;0.0359803549945354;0.0368716865777969;0.0379653871059418;0.0391867905855179;0.0402206256985664;0.0416845157742500;0.0432726368308067;0.0448395796120167;0.0465000607073307;0.0482000969350338;0.0501286424696445;0.0523051992058754;0.0546168871223927;0.0567107088863850;0.0596671998500824;0.0630814060568810;0.0658818855881691;0.0687267109751701;0.0726765841245651;0.0758486911654472;0.0797826200723648;0.0839349254965782;0.0876496508717537;0.0912587568163872;0.0941457822918892;0.0975076556205750];
    Bs = Bs/norm(Bs);
    
    for KKK = 1:sImgChan
        mShadImg(:, :, KKK) = mImg(:, :, KKK)./(mRefImg(:, :, KKK)+eps) *Bs(KKK) .*mask;
    end

    mRefImg = mImg./(mShadImg+eps);
    for KKK = 1:sImgChan
     mRefImg(:, :, KKK) = BilateralFilt2(mRefImg(:, :, KKK),6,[3,0.1]);
    end
    score1 = LMSE(gt_refl,mRefImg, mask)
    score2 = LMSE(mean(gt_shad,3),mean(mShadImg,3), mask)
reset(gpu)
end
