% 可能改动的地方有： Weightmap, 以下标xxxxxxxxx部分。

clc;clear;close all
addpath ./spec2rgb;
addpath('E:\黄谦\20180124');
nameList = {'ali','green_pig','mask', 'piggy_crown', 'spongebob','pumpkin', 'dinosaur', 'piggy_horse', 'hellokitty', 'cap', 'girl', 'fox'};
nameList2 = {};
gpu = gpuDevice(1);
shadmat = zeros(10,10,12);
refmat = zeros(10,10,12);
lista = [155, 722, 677, 907, 754, 863, 859, 652, 1059, 779, 1057];
listb = [166, 724, 677, 910, 753, 844, 808, 654, 1059, 781, 1059];
listc = [230, 636, 373, 672, 416, 595, 885, 531, 592, 968, 580];
listd = [288, 652, 381, 697, 449, 629, 885, 531, 591, 968, 580];
for nameIdx = 1%:length(nameList)-1
    if nameIdx == 5
        continue;
    end
    FILENAME = char(nameList(nameIdx));
    addpath(strcat('E:\黄谦\20180124\',FILENAME));

%% read spectral data
[specImg specImgRGB height width numSpec wl]=readDat('diffuse.dat');specImg = specImg(:, :, 1:3:119);
[gt_shad gt_shadRGB height width numSpec wl]=readDat('shading.dat');gt_shad = gt_shad(:, :, 1:3:119);
gt_shad = imtranslate(gt_shad, [-lista(nameIdx)+listb(nameIdx), -listc(nameIdx)+listd(nameIdx), 0]);
% RGB = spec2rgb(specImg);
% figure, imshow(RGB);    
% RGB = spec2rgb(gt_shad);
% figure, imshow(RGB);   

gt_refl =  specImg./(gt_shad+eps); 
mask = imread('mask.bmp');
mask = mean(mask, 3);
mask = logical(mask);
%groundtruth;

% specImgRGB = im2double(imread('diffuse.bmp'));
% figure, imshow(specImgRGB)
%---------------block--------------
[mask, specImg, gt_shad, gt_refl] = RESIZE(0.5^floor(nameIdx/6)*0.5, mask, specImg, gt_shad, gt_refl);
[Row, Col, Chn] = size(specImg);
K = Chn;
N = Row*Col;

specImg = specImg.*repmat(mask, [1 1 K]);
specImgVec = reshape(specImg, [], K);
specImg = (specImg - min(specImg(:)))/(max(specImg(:))- min(specImg(:)));
gt_shad = (gt_shad - min(gt_shad(:)))/(max(gt_shad(:))- min(gt_shad(:)));

RGB = spec2rgb(specImg);
% figure, imshow(RGB);    


%% Bs, Br
%-----------Br---------------
Jr = 8;
load('450to700_118chan_basis.mat');
coeff = pca(SPECBASIS(:, 1:3:end));
% coeff = pca(specImgVec);
Br = coeff(:, 1:Jr);

% Jr = 3; Br = eye(3);

%-----------Bs---------------
Js = 1;
% Bs = [0.0341521687805653;0.0337826795876026;0.0339604318141937;0.0340989902615547;0.0343906134366989;0.0347923971712589;0.0352059565484524;0.0356034673750401;0.0359618924558163;0.0366208180785179;0.0375161468982697;0.0385161079466343;0.0396221801638603;0.0409136228263378;0.0426436960697174;0.0443429499864578;0.0463294200599194;0.0486194044351578;0.0508781410753727;0.0535426139831543;0.0569861531257629;0.0599890723824501;0.0640699043869972;0.0670775324106216;0.0718319490551949;0.0757113620638847;0.0812561511993408;0.0865379422903061;0.0911443680524826;0.0955437868833542];
Bs = [0.0304664857685566;0.0302258506417274;0.0301303472369909;0.0301258098334074;0.0304012447595596;0.0306360777467489;0.0310054197907448;0.0314002558588982;0.0319754667580128;0.0324994511902332;0.0331317000091076;0.0338466949760914;0.0344399325549603;0.0352818295359612;0.0359803549945354;0.0368716865777969;0.0379653871059418;0.0391867905855179;0.0402206256985664;0.0416845157742500;0.0432726368308067;0.0448395796120167;0.0465000607073307;0.0482000969350338;0.0501286424696445;0.0523051992058754;0.0546168871223927;0.0567107088863850;0.0596671998500824;0.0630814060568810;0.0658818855881691;0.0687267109751701;0.0726765841245651;0.0758486911654472;0.0797826200723648;0.0839349254965782;0.0876496508717537;0.0912587568163872;0.0941457822918892;0.0975076556205750];
% Bs(1:K) = gt_shad(118, 140, 1:K);
Bs = Bs/norm(Bs);

%% Weightmap
H = Normalized(specImg);
beta  = 3e-4;
alpha = 5000;
[Wr, Wd] = weightMap(H, alpha, beta, 0); %xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WrWr = reshape(Wr, [],1);
WdWd = reshape(Wd, [],1);
HVec = reshape(H, [], K);

%% E s-term

    %---------------------sparseBs-----------------------

    Bs_temp = sparse(Bs);
    Bs_temp = repmat({Bs_temp}, N, 1);
    sparseBs = blkdiag(Bs_temp{:});

    NeighSize = 4;
    rowIndex = zeros(1,NeighSize*N*K*2); rowIndex(1:2) = [1,1];
    colIndex = zeros(1,NeighSize*N*K*2);
    
    WscVal = zeros(1,NeighSize*N*K*2);
    Wsc_LVal = zeros(1,NeighSize*N*K*2);
    Vrc_LVal = zeros(1,NeighSize*N*K*2);
    VrcVal = zeros(1,NeighSize*N*K*2);

    idx = 1;        
    for Neighbor=1:NeighSize
    for i = 1:1:N % 行
        coor = [i+1,i+Row,i-1,i-Row];
        p = i;
            if (coor(Neighbor)<N) && (coor(Neighbor)>0)
                q = coor(Neighbor);
                % compute w v
                if Neighbor ==1
                    w = Wd(p);
                elseif Neighbor==2
                    w = Wr(p);
                elseif Neighbor==3
                    w = Wd(q);
                else
                    w = Wr(q);
                end                
                v = 1-w;
                for k = 1:K
                    if idx > 1
                    rowIndex(idx:idx+1) = [rowIndex(idx-1)+1, rowIndex(idx-1)+1];
                    end
                    colIndex(idx:idx+1) = [K*(p-1)+k, K*(q-1)+k];
                    
                    WscVal(idx:idx+1) = [w, -w];
                    Wsc_LVal(idx:idx+1) = [w*specImgVec(q, k), -w*specImgVec(p, k)];
                    Vrc_LVal(idx:idx+1) = [v*specImgVec(q, k), -v*specImgVec(p, k)];
                    VrcVal(idx:idx+1) = [v, -v];
                    
                    idx = idx+2;
                end
            end
        end
    end
    zeroMark = find(rowIndex == 0);
    rowIndex(zeroMark) = []; colIndex(zeroMark) = [];
   
    WscVal(zeroMark) = [];    VrcVal(zeroMark) = [];	Wsc_LVal(zeroMark) = [];    Vrc_LVal(zeroMark) = [];
    
    Wsc = sparse(rowIndex, colIndex, WscVal, N*NeighSize*K, N*K)*sparseBs;
    Vrc_L = sparse(rowIndex, colIndex, Vrc_LVal, N*NeighSize*K, N*K)*sparseBs;
    Wsc = gpuArray(Wsc);
    Vrc_L = gpuArray(Vrc_L);
    
    %---------------------sparseBr-----------------------


    Br_temp = sparse(Br);
    Br_temp = repmat({Br_temp}, N, 1);
    sparseBr = blkdiag(Br_temp{:});
  
    % R estimate
    Wsc_L = sparse(rowIndex, colIndex, Wsc_LVal, N*NeighSize*K, N*K)*sparseBr;
    Vrc = sparse(rowIndex, colIndex, VrcVal, N*NeighSize*K, N*K)*sparseBr;
%     Wsc_L = gpuArray(Wsc_L);
%     Vrc = gpuArray(Vrc);

        % Generic_s
    %---------------------------
    L = speye(N*K);     L = L*sparseBs;
    C = reshape(specImgVec', [], 1);
    L= gpuArray(L);
    C = gpuArray(C);
    
    % Generic_r
    %---------------------------
    L_r = speye(N*K); L_r = L_r*sparseBr;
    C_r = reshape(specImgVec', [], 1);
    L_r= gpuArray(L_r);
    C_r = gpuArray(C_r);

%% Estimate s
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    lambda_sc = 1;%00000;
    lambda_gc = 0;%.1;
    idxI = 0;
    idxJ = 0;
    lambda_rc = 100;
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    %% S estimate
lambda_esti_s = 0.01;
%     Wsc = sparse(rowIndex, colIndex, WscVal, N*NeighSize*K, N*K)*sparseBs;
%     Vrc_L = sparse(rowIndex, colIndex, Vrc_LVal, N*NeighSize*K, N*K)*sparseBs;
%     Gcs = Gc*sparseBs;
        
    b = lambda_esti_s*L'*C;
    Q = lambda_sc*(Wsc'*Wsc) + lambda_rc*(Vrc_L'*Vrc_L) +lambda_esti_s*(L'*L);%+ lambda_gc*(Gc'*Gc);
    disp('Q s-estimate complete!')
    
    tic,
    TOL=1e-8;
    [shadVec,flag,relres,iter]=pcg(Q,b,TOL,2000);
    flag
    relres
    iter
    toc
    
    %     specImgLongVector =specImgVec'; specImgLongVector = sparseBs'*specImgLongVector(:);
    % %     lambda_max = norm(cat(1,lambda_sc*Wsc,lambda_rc*Vrc_L)'*cat(1,-lambda_sc*Wsc*specImgLongVector,-lambda_rc*Vrc_L*specImgLongVector), 'inf');
    % %     lambda = 0.1*lambda_max;
    % %     [shadVec history] = lasso(Wsc+Vrc_L, -Wsc*specImgLongVector-Vrc_L*specImgLongVector, lambda, 1.0, 1.0);
    % 	[shadVec history] = lasso(cat(1,lambda_sc*Wsc,lambda_rc*Vrc_L), cat(1,-lambda_sc*Wsc*specImgLongVector,-lambda_rc*Vrc_L*specImgLongVector), lambda_esti_s, 1.0, 1.0);
    %     shadVec = specImgLongVector+shadVec;
    %     [shadVec history] = lad(cat(1, lambda_sc*Wsc, lambda_rc*Vrc_L, lambda_esti_s*L ),...
    %         cat(1, zeros(N*NeighSize*K, 1), zeros(N*NeighSize*K, 1), lambda_esti_s*C), 1, 1.0);

    
    clear Q b
    a=reshape(shadVec,[Js,N]);
    r=(Bs*a)';
    
    shadImg = zeros(Row, Col, K);
    for k = 1:K
        shadImg(:, :, k) = reshape(gather(r(:, k)), Row, Col).*mask;
    end

    RGB = spec2rgb(gt_shad);
%     figure, imshow(RGB);
    imwrite(RGB, strcat(FILENAME, '_gtshad','.png'));
    RGB = spec2rgb(shadImg);
%     figure, imshow(RGB);
    
    score = LMSE(gt_shad,shadImg, mask)

    imwrite(RGB, strcat(FILENAME, '_shading', num2str(score),'.png'));

    derived_reflImg = specImg./(shadImg+eps);
    score2 = LMSE( gt_refl,derived_reflImg, mask)

    RGB = spec2rgb(derived_reflImg);
%     figure, imshow(RGB, []);
    imwrite(RGB, strcat(FILENAME, '_deriveRef', num2str(score2),'.png'));

%% E r-term
    
    lambda_esti_r = lambda_esti_s;%0.005;
    lambda_data = 1;

    b = lambda_esti_r*L_r'*C_r;
    Q = lambda_sc*(Wsc_L'*Wsc_L) + lambda_rc*(Vrc'*Vrc) +lambda_esti_r*(L_r'*L_r);
    
    shadVec_restore = sparseBs*shadVec;
    Qdata = diag(sparse(gather(shadVec_restore))) * sparseBr;
    Cdata = reshape(specImgVec', [], 1);
    
    Q = Q + lambda_data*(Qdata'*Qdata);
    b = b + lambda_data*Qdata'*Cdata;
    disp('Q r-estimate complete!')

    tic,
    TOL=1e-8;
    [reflVec,flag,relres,iter]=pcg(Q,b,TOL,500);
    flag
    relres
    iter
	toc
    clear Q b
    a=reshape(reflVec,[Jr,N]);
    r=(Br*a)';
    
    reflImage = zeros(Row, Col, K);
    for k = 1:K
        reflImage(:, :, k) = reshape(gather(r(:, k)), Row, Col).*mask;
    end

    RGB = spec2rgb(gt_refl);
%     figure, imshow(RGB);
    imwrite(RGB, strcat(FILENAME, '_gtrefl','.png'));
    RGB = spec2rgb(reflImage);
%     figure, imshow(RGB);
    score3 = LMSE( gt_refl,reflImage, mask)
    imwrite(RGB, strcat(FILENAME, '_refl_', num2str(score3),'.png'));
    
    
%% iteration
% % lambda_rc = 2;
% % lambda_data = 1;%0.01;
% idx = 1;
% %-------------------isolate illumination---------------
% % Bsmat = repmat(Bs(:,1)', Row, 1);
% % Bsmat = repmat(Bsmat, [1, 1, Col]);
% % Bsmat = permute(Bsmat, [1, 3, 2]);
% % gt_shad = gt_shad./Bsmat;
% % gt_shad  =mean(gt_shad, 3);
% % figure, imshow(gt_shad, []);
% % figure
% % LMSE( gt_shad, mean(shadImg,3), mask)
% figure
% while idx < 1000
%     if mod(idx, 2)
%         lambda_data = 0.01;
%     else
%         lambda_data = 1;
%     end
% 
%     
%     
%    E_SUM(idx)=lambda_sc*norm(Wsc*shadVec) + lambda_rc*norm(Vrc_L*shadVec)+...
%          lambda_sc*norm(Wsc_L*reflVec) + lambda_rc*norm(Vrc*reflVec)+...
%          lambda_data*norm((sparseBs*shadVec).*(sparseBr*reflVec)-reshape(specImgVec', [], 1));
%      if idx > 3
%          delta = -E_SUM(idx) +E_SUM(idx-2);
%          if abs(delta) < 0.01
%              break;
%          end
%      end
% %     Solve = {'solve S', 'solve R'};
% %     solveWhat = Solve(mod(idx, 2)+2)
%     
%     if mod(idx, 2)
%         % Qsc
% 
%         Q = lambda_sc*(Wsc'*Wsc);
% 
%         % Qrc
% 
%         Q = Q + lambda_rc*(Vrc_L'*Vrc_L);
%     
%     else
%         % Qsc
% 
%         Q = lambda_sc*(Wsc_L'*Wsc_L);
% 
%         % Qrc
% 
%         Q = Q + lambda_rc*(Vrc'*Vrc);
% 
%     end
% 
%     % Qdata
%     C = reshape(specImgVec', [], 1);
% 
%     if mod(idx, 2)
%     % Edata-s term
%     reflVec_restore = sparseBr * reflVec; % reflVec length: N*Jr
%     Qdata = diag(sparse(reflVec_restore))*sparseBs;
% 
%     else
%     % Edata-r term
%     shadVec_restore = sparseBs * shadVec; % shadVec length: N*Js
%     Qdata = diag(sparse(shadVec_restore))*sparseBr;
% 
%     end
% 
%     Q = Q + lambda_data*(Qdata'*Qdata);
%     b = lambda_data*Qdata'*C;
%     
%     if mod(idx, 2)
%         tic,
%         TOL=1e-6;
%         [shadVec,flag,relres,iter]=pcg(Q,b,TOL,1000);
%         flag
%         relres
%         iter
%         toc
%         
%         a=reshape(shadVec,[Js,N]);
%         r=(Bs*a)';
%         
%         shadImg = zeros(Row, Col, K);
%         for k = 1:K
%             shadImg(:, :, k) = reshape(r(:, k), Row, Col).*mask;
%         end
%         RGB = convRGB(shadImg, 400, 700, (700-400)/(K-1));
%         imshow(RGB);
%         imwrite(RGB, strcat(num2str(idx),'_shading.bmp'));
%         score(idx) = LMSE(gt_shad,shadImg, mask)
%         imwrite(RGB, strcat(FILENAME, '_shading', num2str(score),'.png'));
% 
%     else
%         tic,
%         TOL=1e-6;
%         [reflVec,flag,relres,iter]=pcg(Q,b,TOL,1000);
%         flag
%         relres
%         iter
%         toc
%         
% %         a=reshape(reflVec,[Jr,N]);
% %         r=(Br*a)';
% %         
% %         reflImage = zeros(Row, Col, K);
% %         for k = 1:K
% %             reflImage(:, :, k) = reshape(r(:, k), Row, Col).*mask;
% %         end
%     end
%     idx = idx+1;
% 
% end
    
reset(gpu);
end

